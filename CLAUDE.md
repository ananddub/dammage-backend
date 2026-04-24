# CLAUDE.md

## Commands

```bash
make install    # uv sync
make up         # start MinIO + MongoDB via ../frontend/docker-compose.yml
make dev        # uvicorn --reload on 0.0.0.0:8000
make run        # uvicorn on 0.0.0.0:8000 (LAN-reachable)
make check      # curl / and /reports
make reset      # wipe dammage.reports + MinIO bucket
```

Python 3.12+, managed via `uv`. No linter / formatter / test runner configured.

## Layout

```
backend/
├── src/
│   ├── __init__.py
│   ├── main.py       # FastAPI app + 4 route handlers
│   ├── config.py     # env vars, prompts, keyword map, category metadata
│   ├── ml.py         # model load, tiled YOLO-E inference, severity, annotation
│   └── storage.py    # MinIO + MongoDB init + upload helper
├── best_int8.tflite                     # pothole detector (1-class TFLite, in use)
├── yolov8n-waste-12cls-best_int8.tflite  # legacy 12-class tflite, no longer loaded
├── yoloe-11l-seg.pt                      # auto-downloaded on first request (~100MB)
├── Makefile
├── pyproject.toml
└── uv.lock
```

No auth. CORS fully open.

## Routes

- `GET /` — health + component booleans (`waste_model`, `road_model`, `minio`, `mongo`).
- `POST /report` — multipart: `file`, `lat`, `lng`, optional `email`.
  Runs waste + road detection in parallel. See "Upload flow" below.
- `POST /resolve` — form: `lat`, `lng`, optional `type` (`trash` | `pothole`),
  optional `radius_m`. **Soft-resolves** matching reports (no hard delete).
- `GET /reports` — query: `limit` (default 200), optional `type`,
  `include_resolved` (default `false`). Returns:
  ```json
  [{
    "id": "77.209,28.6139:trash",
    "image": "<MinIO URL>",
    "coordinates": [77.209, 28.6139],
    "time": "ISO",
    "severity_score": 61.07,
    "type": "trash",
    "resolved": false,
    "resolved_at": null
  }]
  ```

## Upload flow (`POST /report`)

1. Coords snapped to `LOCATION_PRECISION` dp (default 4 → ~11 m bucket).
2. Both models run on the image:
   - **Waste** → YOLO-E tiled inference (640 tiles @ 35% overlap + full 1280 pass), merged by canonical category.
   - **Road** → TFLite single pass at 640.
3. **Branch:**
   - **Both empty** → soft-resolve all docs within `CLEANUP_RADIUS_M` (500 m) of
     (lat, lng). `resolved_by: "auto-clean"`. Response: `{cleaned: true, resolved_count: N}`.
   - **Anything detected** → upload original (re-encoded JPEG) + annotated PNG to
     MinIO keyed on coords, then upsert one doc per detected type via
     `replace_one(upsert=True)` matching on deterministic `_id`. Any prior doc of a
     now-missing type at the same spot is soft-resolved (`resolved_by: "auto-stale"`).

## Mongo schema (`dammage.reports`)

Indexes: `2dsphere` on `location`, descending on `time`.

| Field | Shape | Note |
|-------|-------|------|
| `_id` | string `"<lng>,<lat>:<type>"` | Deterministic — same spot + type → same doc |
| `email` | string | `""` if not provided |
| `location` | GeoJSON Point `{type: "Point", coordinates: [lng, lat]}` | Indexed |
| `image` | URL | Annotated PNG (preferred) or original |
| `image_original` | URL | Raw input JPEG |
| `time` | ISODate | Upsert timestamp |
| `type` | `"trash" \| "pothole"` | — |
| `severity_score` | 0–100 | Overall |
| `environmental_impact` | 0–100 | Trash only |
| `detections` | array | Per-box details; polygons stripped before persist |
| `stats` | object | Trash only: coverage %, class/category counts |
| `resolved` | bool | Default `false` on fresh upsert |
| `resolved_at` | ISODate \| null | When flag flipped |
| `resolved_by` | `"auto-clean" \| "auto-stale" \| "manual" \| null` | Why flipped |

Geo cleanup query uses `$geoWithin` + `$centerSphere` — radians = `radius_m / 6_378_100`.

### Soft-delete semantics

No `delete_many` anywhere — all removals are `update_many({...}, {$set: {resolved: true, resolved_at, resolved_by}})`. `/reports` filters `resolved != true` unless `include_resolved=true`. Fresh upserts reset `resolved: false`, effectively reviving a spot.

## MinIO

Bucket `dammage` auto-created with public-read policy. Objects stored under:

```
reports/<lng>,<lat>/input.jpg       ← re-encoded JPEG, overwrite on re-upload
reports/<lng>,<lat>/annotated.png   ← bounding boxes + severity panel
```

Input is **always re-encoded to JPEG** (extension stable across re-uploads). Polygons from YOLO-E segmentation are drawn on the annotated PNG.

`MINIO_PUBLIC_HOST` is the URL **returned to clients** — set to the LAN IP (`192.168.1.3`) so phones/other machines can fetch. Server-to-MinIO calls use loopback via `MINIO_ENDPOINT`.

## Models

### Waste — `yoloe-11l-seg.pt` (YOLO-E large, seg, open-vocabulary)

- Auto-downloaded by ultralytics on first use to backend working directory (~100 MB).
- Bound to 38 `TRASH_PROMPTS` ("plastic bottle", "garbage heap", "cigarette butt", …) via `model.get_text_pe(prompts)` + `model.set_classes(prompts, pe)`.
- Per-detection raw label (e.g. `"plastic bottle"`) mapped to canonical category via `HEURISTIC_KEYWORDS` substring match in priority order. Specific keywords (`cigarette`, `cardboard`) before generic (`trash`, `pile`).
- **Tiled inference** — 640 px tiles at 35 % overlap plus a single 1280 px full-image pass. Merged by `(canonical_category, box IoU > 0.55)`.
- GPU (`cuda:0`) auto-detected; FP16 enabled on CUDA. Typical latency: ~1 s/image on GPU, 30–50 s on CPU.
- `WASTE_CONF=0.08` default (env override) — low for open-vocab recall.

### Road — `best_int8.tflite`

- Single-class pothole detector, TFLite int8.
- Loaded via `ultralytics.YOLO(path, task="detect")`.
- Single pass at 640 px, `ROAD_CONF=0.25`.
- TFLite runtime is `ai-edge-litert`; a `tflite_runtime` shim (`.venv/lib/python3.12/site-packages/tflite_runtime/`) re-exports `ai_edge_litert.interpreter` because upstream `tflite-runtime` has no Python 3.12 wheel.

## Severity formulas

- **Waste (per-detection):** `clamp(area_pct * 3 * (0.5 + 0.5 * pollution) * conf * (0.85 + 0.15 * position_weight), 100)`
  - `position_weight` = 1 at image center, 0 at corners
  - Overall = average × (1 + total_coverage/100), capped at 100
- **Waste (environmental impact):** `(0.4 * pollution + 0.3 * hazard + 0.3 * log-normalised_decomp_years) * 100 * area_factor`
- **Road:** `(min(len * 12, 60) + total_area_pct * 2) * (0.7 + 0.3 * avg_conf)`, capped at 100

`CATEGORY_META` (`config.py`): per-category `decomp_years`, `pollution`, `hazard`, `recyclable`.

## Tuning knobs (env overrides)

| Var | Default | What it does |
|-----|---------|--------------|
| `MINIO_ENDPOINT` | `127.0.0.1:9000` | Server ↔ MinIO connection |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | `admin` / `password123` | Access keys |
| `MINIO_BUCKET` | `dammage` | — |
| `MINIO_PUBLIC_HOST` | `http://192.168.1.3:9000` | Baked into response URLs |
| `MONGO_URI` | `mongodb://127.0.0.1:27017` | — |
| `MONGO_DB` / `MONGO_COLL` | `dammage` / `reports` | — |
| `CLEANUP_RADIUS_M` | `500` | Auto-clean geo radius |
| `LOCATION_PRECISION` | `4` | Rounding dp for spot bucket (4 ≈ 11 m) |
| `WASTE_MODEL` | `yoloe-11l-seg.pt` | Swap the waste weights |
| `WASTE_CONF` | `0.08` | Lower = more recall, more FP |

## Gotchas

1. **Coordinate order** — stored as GeoJSON `[lng, lat]`, not `[lat, lng]`. Leaflet/Mapbox expects `[lat, lng]`; flip on the frontend.
2. **First `/report` call is slow** (~30–60 s) because YOLO-E downloads. Subsequent calls are fast.
3. **Another uvicorn on :8000** from an old process can silently consume frontend requests while you test on a different port — check `ss -tlnp | grep :8000`.
4. **Polygon field `_polygon`** lives on each detection for rendering; stripped before Mongo insert (would bloat docs).
5. **Deterministic `_id`** means changing `LOCATION_PRECISION` mid-database creates orphans — run `make reset` after tuning.
