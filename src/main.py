"""FastAPI app — routes + lifespan. Inference/storage live in ml.py and storage.py."""
from __future__ import annotations

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from pymongo.collection import Collection

from .config import ADMINS_COLL, CLEANUP_RADIUS_M, EARTH_RADIUS_M, LOCATION_PRECISION, MONGO_COLL
from .ml import annotate, load_road, load_waste, run_road, run_waste
from .storage import init_minio, init_mongo, upload

_state: dict = {"waste": None, "road": None, "minio": None, "mongo": None}

# Bound concurrent YOLO inference. Single CPU-only host thrashes badly if N
# torch passes run at once. Requests above the limit queue on the semaphore.
INFERENCE_CONCURRENCY = max(1, int(os.getenv("INFERENCE_CONCURRENCY", "1")))
_INFERENCE_SEM = asyncio.Semaphore(INFERENCE_CONCURRENCY)

# Docs carry a `resolved` boolean (default False). /reports hides anything with
# resolved=True; /resolve and the auto-cleanup paths flip the flag instead of
# running delete_many, so history stays queryable.
NOT_RESOLVED = {"$ne": True}

# Admin-controlled workflow status. `pending` on first upsert.
VALID_STATUSES = {"pending", "acknowledged", "in_progress", "resolved", "rejected"}


def _require_admin(email: str | None) -> str:
    """Gate for mutation routes — looks the caller up in the `admins` collection."""
    normalised = (email or "").strip().lower()
    if not normalised:
        raise HTTPException(403, "admin access required — supply an admin email")
    db = _state.get("mongo")
    if db is None:
        raise HTTPException(503, "admin check unavailable — MongoDB is down")
    if db[ADMINS_COLL].count_documents({"_id": normalised}, limit=1) == 0:
        raise HTTPException(403, f"'{normalised}' is not an admin")
    return normalised


def _admin_emails() -> list[str]:
    """Read current admin roster from Mongo (sorted). Returns [] if mongo is down."""
    db = _state.get("mongo")
    if db is None:
        return []
    return [d["_id"] for d in db[ADMINS_COLL].find({}, {"_id": 1}).sort("_id", 1)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["waste"] = load_waste()
    _state["road"] = load_road()
    _state["minio"] = init_minio()
    _state["mongo"] = init_mongo()
    yield


app = FastAPI(title="Dammage Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


def _read_image(data: bytes) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")


def _encode_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _snap_coords(lng: float, lat: float) -> tuple[float, float]:
    """Round to the spot-precision so GPS jitter maps to the same storage key."""
    return round(lng, LOCATION_PRECISION), round(lat, LOCATION_PRECISION)


def _soft_resolve(coll: Collection, query: dict, source: str) -> int:
    """Flip resolved=True on all docs matching the query. Returns count."""
    query = {**query, "resolved": NOT_RESOLVED}
    now = datetime.now(timezone.utc)
    res = coll.update_many(query, {
        "$set": {"resolved": True, "resolved_at": now, "resolved_by": source}
    })
    return res.modified_count


@app.get("/")
def root():
    return {
        "status": "ok",
        "waste_model": _state["waste"] is not None,
        "road_model": _state["road"] is not None,
        "minio": _state["minio"] is not None,
        "mongo": _state["mongo"] is not None,
        "endpoints": [
            "POST /report", "POST /resolve",
            "GET /reports", "PATCH /reports/{id}",
            "POST /reports/{id}/acknowledge",
            "POST /reports/{id}/start",
            "POST /reports/{id}/resolve",
            "POST /reports/{id}/reject",
            "GET /admins", "GET /admins/{email}", "POST /admins", "DELETE /admins/{email}",
        ],
        "statuses": sorted(VALID_STATUSES),
        "admins": _admin_emails(),
    }


@app.post("/report")
async def report(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lng: float = Form(...),
    email: str = Form(""),
):
    t0 = time.perf_counter()
    db = _state.get("mongo")
    if db is None:
        raise HTTPException(500, "MongoDB not available")

    data = await file.read()
    img = _read_image(data)
    W, H = img.size

    waste_dets, waste_stats, waste_sev, waste_imp = run_waste(_state["waste"], img)
    road_dets, road_sev = run_road(_state["road"], img)

    coll: Collection = db[MONGO_COLL]
    lng_s, lat_s = _snap_coords(lng, lat)

    # ── Cleanup path: nothing detected → soft-resolve the 500 m neighbourhood
    if not waste_dets and not road_dets:
        resolved_count = _soft_resolve(coll, {
            "location": {"$geoWithin": {"$centerSphere": [[lng, lat], CLEANUP_RADIUS_M / EARTH_RADIUS_M]}}
        }, source="auto-clean")
        return {
            "cleaned": True,
            "resolved_count": resolved_count,
            "width": W, "height": H,
            "coordinates": [lng_s, lat_s],
            "waste_detections": 0,
            "road_detections": 0,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── Upload path: overwrite image keyed on rounded coords ("same spot")
    coord_key = f"{lng_s},{lat_s}"
    mc = _state["minio"]
    image_url = upload(mc, f"reports/{coord_key}/input.jpg", _encode_jpeg(img), "image/jpeg")
    annotated_url = upload(
        mc,
        f"reports/{coord_key}/annotated.png",
        annotate(img, waste_dets, road_dets, waste_sev, road_sev),
        "image/png",
    )

    now = datetime.now(timezone.utc)
    geo_point = {"type": "Point", "coordinates": [lng_s, lat_s]}
    common = {
        "email": email,
        "location": geo_point,
        "image": annotated_url or image_url,
        "image_original": image_url,
        "time": now,
        # Re-upload at a previously resolved spot → revive it.
        "resolved": False,
        "resolved_at": None,
        "resolved_by": None,
    }

    # Polygons live on each detection as "_polygon" so annotate() can draw them,
    # but they're hundreds of points each — strip before persisting.
    waste_dets_persist = [
        {k: v for k, v in d.items() if not k.startswith("_")} for d in waste_dets
    ]

    # Deterministic _id per (spot, type) — update_one with $inc preserves counters.
    trash_id = f"{coord_key}:trash"
    pothole_id = f"{coord_key}:pothole"
    on_insert = {"created_at": now, "status": "pending", "status_updated_at": None, "status_updated_by": None}

    inserted: list[str] = []
    if waste_dets:
        coll.update_one(
            {"_id": trash_id},
            {
                "$set": {**common, "type": "trash",
                         "severity_score": waste_sev, "environmental_impact": waste_imp,
                         "detections": waste_dets_persist, "stats": waste_stats},
                "$inc": {"report_count": 1},
                "$setOnInsert": on_insert,
            },
            upsert=True,
        )
        inserted.append("trash")
    if road_dets:
        coll.update_one(
            {"_id": pothole_id},
            {
                "$set": {**common, "type": "pothole",
                         "severity_score": road_sev,
                         "detections": road_dets},
                "$inc": {"report_count": 1},
                "$setOnInsert": on_insert,
            },
            upsert=True,
        )
        inserted.append("pothole")

    # ── Stale-type sweep: if a prior scan at this exact spot flagged a type
    # that isn't in the current scan, the old doc now points at an annotated
    # image that no longer shows that issue. Soft-resolve it.
    stale = [t for t in ("trash", "pothole") if t not in inserted]
    stale_resolved = 0
    if stale:
        stale_resolved = _soft_resolve(coll, {
            "type": {"$in": stale},
            "location.coordinates": [lng_s, lat_s],
        }, source="auto-stale")

    ids = {t: f"{coord_key}:{t}" for t in inserted}

    return {
        "cleaned": False,
        "inserted": inserted,
        "ids": ids,
        "stale_resolved": stale_resolved,
        "coordinates": [lng_s, lat_s],
        "width": W, "height": H,
        "waste_detections": len(waste_dets),
        "road_detections": len(road_dets),
        "waste_severity": waste_sev,
        "road_severity": road_sev,
        "waste_stats": waste_stats,
        "image_url": image_url,
        "annotated_url": annotated_url,
        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.post("/resolve")
def resolve(
    lat: float = Form(...),
    lng: float = Form(...),
    type: str | None = Form(None),
    radius_m: float = Form(CLEANUP_RADIUS_M),
    admin: str = Form(""),
):
    _require_admin(admin)
    db = _state.get("mongo")
    if db is None:
        raise HTTPException(500, "MongoDB not available")
    query: dict = {
        "location": {"$geoWithin": {"$centerSphere": [[lng, lat], radius_m / EARTH_RADIUS_M]}}
    }
    if type in ("trash", "pothole"):
        query["type"] = type
    resolved_count = _soft_resolve(db[MONGO_COLL], query, source="manual")
    return {"resolved": True, "resolved_count": resolved_count, "radius_m": radius_m, "type": type, "admin": admin}


@app.get("/reports")
def list_reports(
    limit: int = 200,
    type: str | None = None,
    status: str | None = None,
    include_resolved: bool = False,
):
    db = _state.get("mongo")
    if db is None:
        raise HTTPException(500, "MongoDB not available")
    query: dict = {}
    if not include_resolved:
        query["resolved"] = NOT_RESOLVED
    if type in ("trash", "pothole"):
        query["type"] = type
    if status in VALID_STATUSES:
        query["status"] = status
    cur = db[MONGO_COLL].find(
        query,
        {
            "_id": 1, "image": 1, "location.coordinates": 1, "time": 1,
            "severity_score": 1, "type": 1, "resolved": 1, "resolved_at": 1,
            "status": 1, "status_updated_at": 1, "status_updated_by": 1,
            "report_count": 1, "created_at": 1,
        },
    ).sort("time", -1).limit(int(limit))
    return [
        {
            "id": str(d.get("_id")),
            "image": d.get("image"),
            "coordinates": d["location"]["coordinates"],
            "time": d["time"].isoformat(),
            "severity_score": d.get("severity_score", 0.0),
            "type": d.get("type"),
            "status": d.get("status", "pending"),
            "status_updated_at": d["status_updated_at"].isoformat() if d.get("status_updated_at") else None,
            "status_updated_by": d.get("status_updated_by"),
            "report_count": int(d.get("report_count", 1)),
            "created_at": d["created_at"].isoformat() if d.get("created_at") else None,
            "resolved": bool(d.get("resolved")),
            "resolved_at": d["resolved_at"].isoformat() if d.get("resolved_at") else None,
        }
        for d in cur
    ]


class StatusUpdate(BaseModel):
    status: str = Field(..., description="New workflow status")
    admin: str | None = Field(default=None, description="Who changed it (optional)")


def _apply_status(report_id: str, new_status: str, admin: str | None) -> dict:
    """Shared path for every status-change route. Updates Mongo."""
    admin_email = _require_admin(admin)
    if new_status not in VALID_STATUSES:
        raise HTTPException(422, f"invalid status; allowed: {sorted(VALID_STATUSES)}")
    db = _state.get("mongo")
    if db is None:
        raise HTTPException(500, "MongoDB not available")

    now = datetime.now(timezone.utc)
    update: dict = {
        "status": new_status,
        "status_updated_at": now,
        "status_updated_by": admin_email,
    }
    # Sync the implicit `resolved` flag with admin-facing status.
    if new_status == "resolved":
        update.update({"resolved": True, "resolved_at": now, "resolved_by": "admin"})
    elif new_status in ("pending", "acknowledged", "in_progress"):
        update.update({"resolved": False, "resolved_at": None, "resolved_by": None})
    # "rejected" leaves `resolved` untouched — admin filter handles visibility.

    coll = db[MONGO_COLL]
    res = coll.update_one({"_id": report_id}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(404, "report not found")
    return {
        "id": report_id,
        "status": new_status,
        "status_updated_at": now.isoformat(),
        "status_updated_by": admin_email,
    }


@app.patch("/reports/{report_id}")
def update_status(report_id: str, body: StatusUpdate):
    return _apply_status(report_id, body.status, body.admin)


# ── Explicit action routes — frontend admin UI buttons map 1:1 ── #
@app.post("/reports/{report_id}/acknowledge")
def ack(report_id: str, admin: str = Form("")):
    return _apply_status(report_id, "acknowledged", admin)


@app.post("/reports/{report_id}/start")
def start(report_id: str, admin: str = Form("")):
    return _apply_status(report_id, "in_progress", admin)


@app.post("/reports/{report_id}/resolve")
def resolve_report(report_id: str, admin: str = Form("")):
    return _apply_status(report_id, "resolved", admin)


@app.post("/reports/{report_id}/reject")
def reject(report_id: str, admin: str = Form("")):
    return _apply_status(report_id, "rejected", admin)


# ─────────────────────── Admin roster CRUD ─────────────────────── #
class AdminCreate(BaseModel):
    email: str = Field(..., description="New admin's email (will be lowercased)")
    name: str | None = Field(default=None, description="Optional display name")
    admin: str = Field(..., description="Existing admin's email — grants auth to add")


def _serialize_admin(d: dict) -> dict:
    return {
        "email": d["_id"],
        "name": d.get("name"),
        "added_at": d["added_at"].isoformat() if d.get("added_at") else None,
        "added_by": d.get("added_by"),
    }


@app.get("/admins")
def list_admins():
    db = _state.get("mongo")
    if db is None:
        raise HTTPException(503, "MongoDB unavailable")
    return [_serialize_admin(d) for d in db[ADMINS_COLL].find({}).sort("_id", 1)]


@app.get("/admins/{email}")
def check_admin(email: str):
    """Verify whether `email` is in the admin roster. 200 with doc or 404."""
    db = _state.get("mongo")
    if db is None:
        raise HTTPException(503, "MongoDB unavailable")
    target = email.strip().lower()
    doc = db[ADMINS_COLL].find_one({"_id": target})
    if doc is None:
        raise HTTPException(404, f"'{target}' is not an admin")
    return _serialize_admin(doc)


@app.post("/admins")
def add_admin(body: AdminCreate):
    actor = _require_admin(body.admin)
    new_email = body.email.strip().lower()
    if not new_email or "@" not in new_email:
        raise HTTPException(422, "invalid email")
    now = datetime.now(timezone.utc)
    coll = _state["mongo"][ADMINS_COLL]
    coll.update_one(
        {"_id": new_email},
        {
            "$set": {"name": body.name},
            "$setOnInsert": {"added_at": now, "added_by": actor},
        },
        upsert=True,
    )
    return _serialize_admin(coll.find_one({"_id": new_email}))


@app.delete("/admins/{email}")
def remove_admin(email: str, admin: str = ""):
    actor = _require_admin(admin)
    target = email.strip().lower()
    if target == actor:
        raise HTTPException(409, "admins cannot remove themselves")
    coll = _state["mongo"][ADMINS_COLL]
    if coll.count_documents({}) <= 1:
        raise HTTPException(409, "cannot remove the last remaining admin")
    res = coll.delete_one({"_id": target})
    if res.deleted_count == 0:
        raise HTTPException(404, "admin not found")
    return {"removed": target, "by": actor}
