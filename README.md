# Dammage Backend

FastAPI + TFLite waste & pothole detection. Stores reports in MongoDB with GeoJSON
location; uploads images to MinIO. Auto-resolves entries within 500 m when a new
clean photo of the same spot is uploaded.

## Quick start

```bash
make install     # uv sync
make up          # docker compose up MinIO + MongoDB
make run         # uvicorn on 0.0.0.0:8000
make check       # health + recent reports
```

## Routes

| Method | Path        | Description |
|--------|-------------|-------------|
| GET    | `/`         | Health — model / MinIO / Mongo status |
| POST   | `/report`   | Upload image + lat/lng; runs trash + pothole detect, stores or auto-resolves |
| POST   | `/resolve`  | Explicit cleanup by (lat, lng) without uploading an image |
| GET    | `/reports`  | Pull saved reports: `{image, coordinates, time, severity_score, type}` |

`POST /report` form fields: `file` (image), `lat`, `lng`, optional `email`.
If nothing is detected → deletes all reports within `CLEANUP_RADIUS_M` (default 500 m).
Otherwise → inserts one document per detected type (`trash` / `pothole`).

`POST /resolve` form fields: `lat`, `lng`, optional `type` (`trash` | `pothole`),
optional `radius_m`. Deletes matching reports.

`GET /reports` query params: `limit` (default 200), optional `type`.

## Layout

```
backend/
├── src/
│   ├── __init__.py
│   ├── main.py     # FastAPI app + route handlers
│   ├── config.py   # env vars, category/class metadata, colors
│   ├── ml.py       # model load, inference, severity scoring, annotation
│   └── storage.py  # MinIO + MongoDB init + upload helpers
├── best_int8.tflite                     # pothole detector (1 class)
├── yolov8n-waste-12cls-best_int8.tflite # waste detector (12 classes)
├── Makefile
└── pyproject.toml
```

## Models

- **Waste**: YOLOv8n int8 TFLite, 12 classes — battery, biological, brown-glass,
  cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass.
  Mapped in `config.py::CLASS_CATEGORY` to 8 categories (plastic/paper/glass/metal/
  organic/hazardous/textile/mixed) with pollution, hazard, and decomposition metadata.
- **Road**: single-class TFLite pothole detector.

Both load via `ultralytics.YOLO(task="detect")`. TFLite runtime is provided by
`ai-edge-litert` with a small `tflite_runtime` shim because upstream
`tflite-runtime` doesn't publish wheels for Python 3.12.

## Env vars (optional)

| Var                   | Default                             |
|-----------------------|-------------------------------------|
| `MINIO_ENDPOINT`      | `127.0.0.1:9000`                    |
| `MINIO_ROOT_USER`     | `admin`                             |
| `MINIO_ROOT_PASSWORD` | `password123`                       |
| `MINIO_BUCKET`        | `dammage`                           |
| `MINIO_PUBLIC_HOST`   | `http://192.168.1.3:9000`           |
| `MONGO_URI`           | `mongodb://127.0.0.1:27017`         |
| `MONGO_DB`            | `dammage`                           |
| `MONGO_COLL`          | `reports`                           |
| `CLEANUP_RADIUS_M`    | `500`                               |
