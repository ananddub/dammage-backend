FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libgl1 libsm6 libxrender1 libxext6 curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Dependencies first — cache layer. `type=cache` mount keeps uv's wheel cache
# outside the image, so even `docker builder prune -f` (without -a) preserves
# the downloaded wheels across rebuilds.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Swap the default CUDA torch wheels (pulled via ultralytics) for the CPU-only
# build — the server has no GPU. Drops the image by ~2 GB so BuildKit's
# "exporting to layers" step actually completes in a reasonable time.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        --force-reinstall \
        torch torchvision

# ultralytics's TFLite backend imports `tflite_runtime`, but upstream doesn't
# publish wheels for Python 3.12. Alias to ai-edge-litert (already in deps).
RUN /app/.venv/bin/python - <<'PY'
import os, site
p = site.getsitepackages()[0]
os.makedirs(f"{p}/tflite_runtime", exist_ok=True)
with open(f"{p}/tflite_runtime/__init__.py", "w") as f:
    f.write("from ai_edge_litert import __version__\n")
with open(f"{p}/tflite_runtime/interpreter.py", "w") as f:
    f.write("from ai_edge_litert.interpreter import Interpreter, load_delegate\n")
PY

# YOLO-E's text encoder pulls ultralytics's CLIP fork from Git at first use.
# Install explicitly so build is deterministic and offline-capable at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python \
        "git+https://github.com/ultralytics/CLIP.git"

# Pre-download YOLO-E weights + CLIP text encoder so first /report isn't a cold-start.
RUN /app/.venv/bin/python -c "from ultralytics import YOLO; m = YOLO('yoloe-11l-seg.pt'); m.get_text_pe(['trash'])"

COPY src ./src
COPY best_int8.tflite ./

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
