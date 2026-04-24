# Dammage Backend

FastAPI service for AI-powered civic infrastructure reporting. A citizen uploads
a geotagged photo; the backend runs **two models** on it (open-vocabulary trash
detection + pothole detection), stores the result in MongoDB, and pushes the
image to MinIO.

- **Image upload** → MinIO, keyed on snapped GPS coords so re-uploads at the
  same spot **overwrite** the previous image.
- **Deterministic Mongo `_id`** `"<lng>,<lat>:<type>"` → same spot, same ID,
  upsert preserves counters and status across re-uploads.
- **Counter** — `report_count` increments every time someone re-reports the
  same spot-type.
- **Admin workflow status** — `pending → acknowledged → in_progress → resolved | rejected`.
- **Soft-delete via `resolved` flag** — nothing is ever hard-deleted; history
  stays queryable.
- **Auto-resolve on clean re-scan** — if the ML pipeline finds nothing in a
  newly uploaded image, all reports within 500 m of that point are flipped to
  resolved.
- **Two-tier access** — citizens can upload and view everything; only
  allow-listed admin emails can change status or resolve areas.

---

## Quick start

```bash
make install     # uv sync
make up          # start MinIO + MongoDB via ../frontend/docker-compose.yml
make dev         # uvicorn --reload on 0.0.0.0:8000
make check       # hit / and /reports
make reset       # wipe dammage.reports + MinIO bucket
```

Python 3.12+, `uv`-managed.

---

## Stack & layout

```
backend/
├── src/
│   ├── main.py          # FastAPI app + 8 route handlers
│   ├── config.py        # env vars, prompts, keyword map, category metadata
│   ├── ml.py            # model load, tiled YOLO-E inference, severity, annotation
│   └── storage.py       # MinIO + MongoDB init + helpers
├── best_int8.tflite                      # pothole detector (1-class TFLite, in use)
├── yolov8n-waste-12cls-best_int8.tflite  # legacy 12-class tflite, no longer loaded
├── yoloe-11l-seg.pt                      # auto-downloaded on first request (~100 MB)
├── Dockerfile                            # python:3.12-slim + uv + pre-download YOLO-E/CLIP
├── docker-compose.prod.yml               # mongo + minio + backend
├── Makefile
├── pyproject.toml
└── uv.lock
```

CORS fully open. No session/JWT auth — the admin gate is just an email
allow-list enforced on mutation routes.

---

## Access control

Two tiers of caller:

| Capability | Citizen | Admin |
|------------|:-------:|:-----:|
| `GET /` — health + admin roster | ✓ | ✓ |
| `GET /reports` — list / view all | ✓ | ✓ |
| `POST /report` — upload a photo | ✓ | ✓ |
| `POST /resolve` — clear an area without an image | — | ✓ |
| `PATCH /reports/{id}` — change status | — | ✓ |
| `POST /reports/{id}/{acknowledge\|start\|resolve\|reject}` | — | ✓ |

Admin check:

- Mutation routes require an `admin` field (form for `POST`/`action`, JSON for `PATCH`).
- Its value must be in `ADMIN_EMAILS` (env var, comma-separated, case-insensitive).
- Missing or non-matching email → **`403`** with message `"admin access required — supply a valid admin email"`.
- Default `ADMIN_EMAILS=duanand6@gmail.com`. Override at boot to add more:
  `ADMIN_EMAILS="duanand6@gmail.com,ops@team.com"`.

The admin roster is advertised on `GET /` so frontends can decide whether to
render edit controls:

```ts
const { admins } = await (await fetch("/")).json()
const isAdmin = admins.includes(currentUser.email.toLowerCase())
```

There is **no auth header or session** — the `admin` field is trusted as-is.
Fine for a hackathon but you'd layer real auth (magic-link, OAuth, JWT) on top
for anything production.

---

## API reference

Base URL: `http://<host>:8000`.

### `GET /` — health

```json
{
  "status": "ok",
  "waste_model": true,
  "road_model": true,
  "minio": true,
  "mongo": true,
  "endpoints": [
    "POST /report", "POST /resolve",
    "GET /reports", "PATCH /reports/{id}",
    "POST /reports/{id}/acknowledge",
    "POST /reports/{id}/start",
    "POST /reports/{id}/resolve",
    "POST /reports/{id}/reject"
  ],
  "statuses": ["acknowledged", "in_progress", "pending", "rejected", "resolved"],
  "admins": ["duanand6@gmail.com"]
}
```

Boolean per component tells you what's wired up. Missing a model or a broker
still returns 200; calls that need that component will return 500.

---

### `POST /report` — upload a photo at a location

**Form fields**

| Field | Required | Notes |
|-------|----------|-------|
| `file` | ✅ | image file — any format Pillow reads (JPEG / PNG / WEBP / HEIC) |
| `lat`  | ✅ | latitude float |
| `lng`  | ✅ | longitude float |
| `email`| ❌ | reporter identity, `""` if omitted |

**What happens**

1. Coords are snapped to `LOCATION_PRECISION` dp (default 4 → ~11 m bucket) so
   GPS jitter maps to the same spot.
2. **Waste** (open-vocab YOLO-E) and **Road** (pothole TFLite) run sequentially
   under a semaphore (default concurrency 1).
3. **Branch:**
   - Neither detector fires ⇒ **auto-clean path**. Every un-resolved doc
     within `CLEANUP_RADIUS_M` (default 500 m) of (lat, lng) is flipped to
     `resolved: true, resolved_by: "auto-clean"`. No image is stored.
   - Anything detected ⇒ **upload path**. The raw image is re-encoded as JPEG
     and written to MinIO at `reports/<lng>,<lat>/input.jpg`. An annotated PNG
     (bounding boxes + severity panel) is written at `annotated.png`. Up to two
     Mongo docs are upserted (one per detected type) with deterministic `_id`.
     `report_count` is incremented. Any previously-resolved doc at the spot is
     revived. Stale types (e.g. trash was seen last time but only pothole now)
     are soft-resolved with `resolved_by: "auto-stale"`.

**Response — auto-clean path**

```json
{
  "cleaned": true,
  "resolved_count": 2,
  "width": 800, "height": 600,
  "coordinates": [77.5, 28.5],
  "waste_detections": 0,
  "road_detections": 0,
  "processing_time_ms": 1120.4
}
```

**Response — upload path**

```json
{
  "cleaned": false,
  "inserted": ["trash", "pothole"],
  "ids": {
    "trash": "77.5,28.5:trash",
    "pothole": "77.5,28.5:pothole"
  },
  "stale_resolved": 0,
  "coordinates": [77.5, 28.5],
  "width": 1280, "height": 960,
  "waste_detections": 4,
  "road_detections": 1,
  "waste_severity": 58.4,
  "road_severity": 42.5,
  "waste_stats": {
    "total_detections": 4,
    "total_coverage_pct": 18.2,
    "class_counts": { "plastic bottle": 2, "cardboard": 2 },
    "category_counts": { "plastic": 2, "paper": 2 }
  },
  "image_url": "http://<public>:9000/dammage/reports/77.5,28.5/input.jpg",
  "annotated_url": "http://<public>:9000/dammage/reports/77.5,28.5/annotated.png",
  "processing_time_ms": 2840.1
}
```

---

### `GET /reports` — list reports

**Query params**

| Param | Default | Notes |
|-------|---------|-------|
| `limit` | `200` | Max rows returned |
| `type` | _(none)_ | `trash` or `pothole` — filter |
| `status` | _(none)_ | one of the workflow statuses |
| `include_resolved` | `false` | show rows where `resolved=true` |

**Response** (array, sorted by `time` desc)

```json
[
  {
    "id": "77.5,28.5:trash",
    "image": "http://<public>:9000/dammage/reports/77.5,28.5/annotated.png",
    "coordinates": [77.5, 28.5],
    "time": "2026-04-25T06:42:01.447000",
    "severity_score": 58.4,
    "type": "trash",
    "status": "in_progress",
    "status_updated_at": "2026-04-25T07:10:22.914000",
    "status_updated_by": "anand",
    "report_count": 3,
    "created_at": "2026-04-24T22:15:09.120000",
    "resolved": false,
    "resolved_at": null
  }
]
```

Coordinates come out as **GeoJSON order `[lng, lat]`** — flip for Leaflet /
Mapbox.

---

### `POST /resolve` — soft-resolve by coords (no image)

Admin / ops button for "I cleaned this area, hide reports without me
re-uploading a photo." **Admin-only.**

**Form fields**

| Field | Required | Notes |
|-------|----------|-------|
| `lat`, `lng` | ✅ | center of the area to resolve |
| `admin` | ✅ | admin email — must be in `ADMIN_EMAILS` or `403` |
| `type` | ❌ | `trash` or `pothole`; omit to clear both |
| `radius_m` | ❌ | default `500` |

**Response**

```json
{ "resolved": true, "resolved_count": 4, "radius_m": 500, "type": null, "admin": "duanand6@gmail.com" }
```

Flips `resolved=true, resolved_by: "manual"` on each match.

---

### Status change endpoints

**Admin-only.** All 4 action routes accept `admin` as a **required** form
field; `PATCH` takes it in the JSON body. The email must be in the
`admins` collection or the route returns `403`. On success they write
`status_updated_at` + `status_updated_by` and sync the `resolved` flag.

| Method | Path | New status |
|--------|------|-----------|
| `POST` | `/reports/{id}/acknowledge` | `acknowledged` |
| `POST` | `/reports/{id}/start` | `in_progress` |
| `POST` | `/reports/{id}/resolve` | `resolved` (also flips `resolved=true`) |
| `POST` | `/reports/{id}/reject` | `rejected` |
| `PATCH` | `/reports/{id}` | any (JSON body `{"status":"…","admin":"…"}`) |

**Response** (shared shape)

```json
{
  "id": "77.5,28.5:trash",
  "status": "in_progress",
  "status_updated_at": "2026-04-25T07:10:22.914000+00:00",
  "status_updated_by": "anand"
}
```

**Side effects**

| New status | `resolved` flag |
|------------|-----------------|
| `pending`, `acknowledged`, `in_progress` | set to `false` (revived to active) |
| `resolved` | set to `true`, `resolved_by: "admin"` |
| `rejected` | untouched (admin filter controls visibility) |

Missing / non-admin `admin` → `403`.
Unknown status → `422` with the allowed list.
Unknown `id` → `404`.

---

## MongoDB schema (`dammage.reports`)

**Indexes:** `2dsphere` on `location`, descending on `time`.

| Field | Type | Populated by |
|-------|------|--------------|
| `_id` | string `"<lng>,<lat>:<type>"` | derived from snapped coords + type |
| `email` | string | upload form field (`""` if absent) |
| `location` | GeoJSON Point `{type: "Point", coordinates: [lng, lat]}` | snapped coords |
| `image` | URL | annotated PNG (preferred) or original |
| `image_original` | URL | raw input JPEG |
| `time` | Date | last upsert |
| `created_at` | Date | first insert (`$setOnInsert`) |
| `type` | `"trash" \| "pothole"` | per-detection |
| `severity_score` | 0–100 | overall per-type severity |
| `environmental_impact` | 0–100 | trash only |
| `detections` | array | per-box details; polygons stripped before persist |
| `stats` | object | trash only — class / category counts, coverage % |
| `report_count` | int | `$inc: 1` on every upsert at same _id |
| `status` | string | admin workflow — default `"pending"` |
| `status_updated_at` | Date \| null | last PATCH / action route |
| `status_updated_by` | string \| null | admin form field |
| `resolved` | bool | soft-delete flag — default `false` |
| `resolved_at` | Date \| null | when flag flipped |
| `resolved_by` | `"auto-clean" \| "auto-stale" \| "manual" \| "admin" \| null` | why it flipped |

### Soft-delete semantics

No `delete_many` anywhere. All removals are `update_many({…}, {$set: {resolved: true, …}})`.
`/reports` filters `resolved: {$ne: true}` by default. Fresh upserts reset
`resolved: false`, effectively **reviving** a resolved spot.

### Geo cleanup

```js
coll.updateMany(
  { "location": { "$geoWithin": { "$centerSphere": [[lng, lat], radius_m / 6378100] } },
    "resolved": { "$ne": true } },
  { "$set": { "resolved": true, "resolved_at": now, "resolved_by": "auto-clean" } }
)
```

---

## MinIO layout

Bucket `dammage`, public-read policy auto-set on first boot. Objects keyed on
snapped coordinates:

```
reports/<lng>,<lat>/input.jpg       ← re-encoded JPEG, stable extension, overwrite on re-upload
reports/<lng>,<lat>/annotated.png   ← bounding boxes / polygons + severity panel
```

`MINIO_PUBLIC_HOST` is the URL **returned to clients** — point it at the LAN
or public IP so browsers/phones can fetch. Backend-internal calls use
`MINIO_ENDPOINT` (loopback or Docker-network hostname).

---

## Models

### Waste — `yoloe-11l-seg.pt` (YOLO-E large, seg, open-vocabulary)

- Auto-downloaded by ultralytics on first use (~100 MB to cwd / build layer).
- Bound to 38 `TRASH_PROMPTS` ("plastic bottle", "garbage heap", "cigarette
  butt", …) via `model.get_text_pe(prompts)` + `model.set_classes(prompts, pe)`.
- Each detection's raw label mapped to a canonical category via
  `HEURISTIC_KEYWORDS` substring match. Specific keywords win over generic.
- Default pipeline: SAHI-style tiled inference — 640 px tiles at 35 % overlap
  plus one full-image pass at 1280 px, merged by (category, box IoU > 0.55).
  `WASTE_TILED=false` env to disable tiling (useful on CPU-only hosts; single
  1280 pass instead).
- GPU (`cuda:0`) auto-detected with FP16 when available.
- `WASTE_CONF=0.08` default (low recall threshold suits open-vocab).

### Road — `best_int8.tflite`

- Single-class pothole detector, TFLite int8, ~12 MB.
- Loaded via `ultralytics.YOLO(path, task="detect")`.
- Single pass at 640 px, `ROAD_CONF=0.25`.
- TFLite runtime is `ai-edge-litert`. A `tflite_runtime` shim lives at
  `.venv/lib/python3.12/site-packages/tflite_runtime/` because upstream
  `tflite-runtime` publishes no Python 3.12 wheel.

---

## Severity formulas

**Waste — per-detection:**

```
severity = clamp(area_pct * 3 * (0.5 + 0.5 * pollution) * conf *
                 (0.85 + 0.15 * position_weight), 100)
```

- `position_weight` = 1 at image centre, 0 at corners.

**Waste — overall:** average per-detection severity × `(1 + coverage/100)`,
capped at 100.

**Environmental impact:**

```
impact = (0.4 * pollution + 0.3 * hazard + 0.3 * log1p(decomp_years) / log1p(1e6))
         * 100 * (0.5 + 0.5 * min(area_pct / 20, 1))
```

**Road:**

```
severity = (min(len(dets) * 12, 60) + total_area_pct * 2) *
           (0.7 + 0.3 * avg_conf)
```
Capped at 100.

Category metadata (`decomp_years`, `pollution`, `hazard`, `recyclable`) lives in
`config.py::CATEGORY_META`.

---

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `MINIO_ENDPOINT` | `127.0.0.1:9000` | backend ↔ MinIO connection |
| `MINIO_ROOT_USER` | `admin` | MinIO access key |
| `MINIO_ROOT_PASSWORD` | `password123` | MinIO secret key |
| `MINIO_BUCKET` | `dammage` | bucket name |
| `MINIO_PUBLIC_HOST` | `http://192.168.1.3:9000` | **baked into response URLs** |
| `MONGO_URI` | `mongodb://127.0.0.1:27017` | Mongo DSN |
| `MONGO_DB` | `dammage` | database |
| `MONGO_COLL` | `reports` | collection |
| `CLEANUP_RADIUS_M` | `500` | auto-clean geo radius |
| `LOCATION_PRECISION` | `4` | rounding dp for spot key (4 ≈ 11 m) |
| `ADMIN_EMAILS` | `duanand6@gmail.com` | comma-separated allow-list for mutation routes |
| `WASTE_MODEL` | `yoloe-11l-seg.pt` | override waste weights |
| `WASTE_CONF` | `0.08` | waste detection threshold |
| `WASTE_TILED` | `true` | disable on CPU-only hosts for speed |
| `INFERENCE_CONCURRENCY` | `1` | max concurrent model passes |

---

## Docker deploy

```bash
# production compose — mongo + minio + backend
docker compose -f docker-compose.prod.yml up -d --build
```

Exposes (override via compose ports block):

- `8000` — backend API
- `9000`, `9001` — MinIO API + console

MongoDB is intentionally **not** exposed publicly in the prod compose — only
reachable on the Docker network.

---

## Gotchas

- **Coord order** — Mongo stores `[lng, lat]` (GeoJSON). Leaflet/Mapbox want
  `[lat, lng]`. Flip on the client.
- **First `/report` call is slow** (~30–60 s) because YOLO-E + CLIP weights
  download on cold start. Pre-downloaded in the `Dockerfile` build stage.
- **Public repo credentials** — the committed compose files hardcode the
  MinIO root user / password. Rotate after hackathon; move to
  `${VAR}` + `.env.example`.
- **`_polygon` is transient** — segmentation polygons are attached to
  detections for annotation drawing, but stripped before Mongo persist to keep
  docs small.
- **`LOCATION_PRECISION` drift** — changing it mid-database orphans existing
  `_id`s (snapping changes). Run `make reset` after retuning.
- **Rogue second uvicorn** — if a stray backend is already listening on `:8000`
  your frontend will hit the wrong one; check `ss -tlnp | grep :8000`.
