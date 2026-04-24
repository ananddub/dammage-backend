"""Constants, env vars, and domain metadata. No imports of app code."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ───────────────────────── Model Config ───────────────────────── #
# Open-vocabulary YOLO-E large seg. Auto-downloads on first YOLO() call (~100MB).
WASTE_MODEL = os.getenv("WASTE_MODEL", "yoloe-11l-seg.pt")
ROAD_TFLITE = BASE_DIR / "best_int8.tflite"

# Road model inference
ROAD_CONF = 0.25
ROAD_IOU = 0.45
ROAD_IMGSZ = 640
ROAD_MAX_DET = 300

# Waste — open-vocab, low conf, tiled like the reference notebook
WASTE_CONF = float(os.getenv("WASTE_CONF", "0.08"))
WASTE_IOU = 0.45
WASTE_MAX_DET = 300
WASTE_TILE_SIZE = 640
WASTE_TILE_OVERLAP = 0.35
WASTE_FULL_IMGSZ = 1280
WASTE_MERGE_IOU = 0.55
WASTE_MAX_IMAGE_DIM = 4096  # resize beyond this before inference
# Tiling trades latency for small-object recall. Disable on CPU-only hosts.
WASTE_TILED = os.getenv("WASTE_TILED", "true").lower() == "true"

# Open-vocabulary prompts passed to YOLO-E.
TRASH_PROMPTS = [
    "plastic bottle", "plastic bag", "plastic cup", "plastic container",
    "plastic wrapper", "plastic food container", "plastic packaging",
    "plastic straw", "plastic lid",
    "paper", "paper cup", "paper bag", "cardboard", "cardboard box",
    "crumpled paper", "newspaper",
    "aluminum can", "tin can", "metal can", "aluminum foil",
    "glass bottle", "broken glass", "glass jar",
    "food waste", "rotten food", "banana peel",
    "cigarette butt",
    "trash pile", "garbage heap", "pile of trash", "mound of garbage",
    "trash", "garbage", "litter", "waste", "rubbish",
    "wrapper", "food wrapper",
]

# Order matters — earlier keywords win. Maps prompt substring → canonical category.
HEURISTIC_KEYWORDS: list[tuple[str, str]] = [
    ("cigarette", "hazardous"),
    ("battery",   "hazardous"),
    ("cardboard", "paper"),
    ("newspaper", "paper"),
    ("carton",    "paper"),
    ("paper",     "paper"),
    ("styrofoam", "plastic"),
    ("foam",      "plastic"),
    ("plastic",   "plastic"),
    ("wrapper",   "plastic"),
    ("bag",       "plastic"),
    ("glass",     "glass"),
    ("aluminum",  "metal"),
    ("tin",       "metal"),
    ("can",       "metal"),
    ("metal",     "metal"),
    ("food",      "organic"),
    ("peel",      "organic"),
    ("rotten",    "organic"),
    ("clothes",   "textile"),
    ("shoes",     "textile"),
    # Generic fallbacks — match last so specific items win first
    ("trash",     "mixed"),
    ("garbage",   "mixed"),
    ("litter",    "mixed"),
    ("rubbish",   "mixed"),
    ("waste",     "mixed"),
    ("pile",      "mixed"),
    ("heap",      "mixed"),
    ("mound",     "mixed"),
]

# ───────────────────────── MinIO ───────────────────────── #
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "password123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "dammage")
MINIO_PUBLIC_HOST = os.getenv("MINIO_PUBLIC_HOST", "http://192.168.1.3:9000")

# ───────────────────────── Mongo ───────────────────────── #
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = os.getenv("MONGO_DB", "dammage")
MONGO_COLL = os.getenv("MONGO_COLL", "reports")

# ───────────────────────── Geo cleanup ───────────────────────── #
CLEANUP_RADIUS_M = float(os.getenv("CLEANUP_RADIUS_M", "500"))
EARTH_RADIUS_M = 6_378_100.0
LOCATION_PRECISION = int(os.getenv("LOCATION_PRECISION", "4"))

# ───────────────────────── Environmental metadata ───────────────────────── #
CATEGORY_META = {
    "plastic":    {"decomp_years": 450,       "pollution": 0.85, "hazard": 0.75, "recyclable": True},
    "paper":      {"decomp_years": 1,         "pollution": 0.20, "hazard": 0.10, "recyclable": True},
    "glass":      {"decomp_years": 1_000_000, "pollution": 0.55, "hazard": 0.85, "recyclable": True},
    "metal":      {"decomp_years": 80,        "pollution": 0.40, "hazard": 0.50, "recyclable": True},
    "organic":    {"decomp_years": 1,         "pollution": 0.20, "hazard": 0.10, "recyclable": False},
    "hazardous":  {"decomp_years": 100,       "pollution": 0.95, "hazard": 0.95, "recyclable": False},
    "textile":    {"decomp_years": 40,        "pollution": 0.45, "hazard": 0.25, "recyclable": True},
    "mixed":      {"decomp_years": 50,        "pollution": 0.60, "hazard": 0.50, "recyclable": False},
}

CATEGORY_COLORS = {
    "plastic":   (59, 130, 246),
    "paper":     (251, 191, 36),
    "glass":     (20, 184, 166),
    "metal":     (148, 163, 184),
    "organic":   (34, 197, 94),
    "hazardous": (239, 68, 68),
    "textile":   (168, 85, 247),
    "mixed":     (217, 119, 6),
    "road":      (220, 38, 38),
}

ROAD_LABEL = "Pothole"
