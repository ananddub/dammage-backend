"""Model loading, inference, severity scoring, and annotation drawing.

Waste → open-vocabulary YOLO-E large seg with tiled inference (matches the
reference notebook). Road → single-class TFLite pothole detector.
"""
from __future__ import annotations

import io
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .config import (
    CATEGORY_COLORS,
    CATEGORY_META,
    HEURISTIC_KEYWORDS,
    ROAD_CONF,
    ROAD_IMGSZ,
    ROAD_IOU,
    ROAD_LABEL,
    ROAD_MAX_DET,
    ROAD_TFLITE,
    TRASH_PROMPTS,
    WASTE_CONF,
    WASTE_FULL_IMGSZ,
    WASTE_IOU,
    WASTE_MAX_DET,
    WASTE_MAX_IMAGE_DIM,
    WASTE_MERGE_IOU,
    WASTE_MODEL,
    WASTE_TILED,
    WASTE_TILE_OVERLAP,
    WASTE_TILE_SIZE,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE.startswith("cuda")


# ───────────────────────── Model loading ───────────────────────── #
def load_waste() -> YOLO | None:
    """Load YOLO-E and bind the open-vocabulary trash prompts.

    `WASTE_MODEL` can be a bare filename — ultralytics auto-downloads known
    model names (like `yoloe-11l-seg.pt`) to the working directory on first
    use (~100 MB).
    """
    print(f"[model] loading waste → {WASTE_MODEL} on {DEVICE} (auto-download if needed)")
    try:
        m = YOLO(str(WASTE_MODEL))
    except Exception as e:
        print(f"[model] waste load failed: {type(e).__name__}: {e}")
        return None
    try:
        text_pe = m.get_text_pe(TRASH_PROMPTS)
        m.set_classes(TRASH_PROMPTS, text_pe)
    except Exception as e:
        print(f"[model] get_text_pe unavailable ({e}); falling back to set_classes")
        try:
            m.set_classes(TRASH_PROMPTS)
        except Exception as e2:
            print(f"[model] set_classes failed: {e2}")
            return None
    print(f"[model] waste ready · {len(TRASH_PROMPTS)} prompts · half={USE_HALF}")
    return m


def load_road() -> YOLO | None:
    path = Path(ROAD_TFLITE)
    if not path.exists():
        print(f"[model] road tflite missing → {path.name}")
        return None
    print(f"[model] loading road  → {path.name}")
    return YOLO(str(path), task="detect")


# ───────────────────────── Generic helpers ───────────────────────── #
def _iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    ba = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    return inter / (aa + ba - inter + 1e-9)


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _resolve_category(raw_label: str) -> tuple[str, dict]:
    low = raw_label.lower()
    for kw, canon in HEURISTIC_KEYWORDS:
        if kw in low:
            return canon, CATEGORY_META[canon]
    return "mixed", CATEGORY_META["mixed"]


def _position_weight(cx: float, cy: float, w: int, h: int) -> float:
    dx, dy = (cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2)
    return max(0.0, 1.0 - (dx * dx + dy * dy) ** 0.5)


def _waste_severity(area_pct: float, conf: float, meta: dict, pw: float) -> float:
    base = min(area_pct * 3.0, 60.0)
    return round(min(base * (0.5 + 0.5 * meta["pollution"]) * conf * (0.85 + 0.15 * pw), 100.0), 2)


def _waste_impact(area_pct: float, meta: dict) -> float:
    decomp_norm = min(np.log1p(meta["decomp_years"]) / np.log1p(1_000_000), 1.0)
    score = (0.4 * meta["pollution"] + 0.3 * meta["hazard"] + 0.3 * decomp_norm) * 100
    return round(score * (0.5 + 0.5 * min(area_pct / 20.0, 1.0)), 2)


def _road_severity(detections: list, img_area: float) -> float:
    if not detections:
        return 0.0
    total_area = sum(
        max(0.0, d["box"]["x2"] - d["box"]["x1"]) * max(0.0, d["box"]["y2"] - d["box"]["y1"])
        for d in detections
    )
    area_pct = (total_area / img_area) * 100.0
    avg_conf = sum(d["confidence"] for d in detections) / len(detections)
    base = min(len(detections) * 12.0, 60.0)
    return round(min((base + area_pct * 2.0) * (0.7 + 0.3 * avg_conf), 100.0), 2)


# ───────────────────────── Road pipeline (tflite, single pass) ───────────────────────── #
def _predict_road(model: YOLO, img: Image.Image) -> list:
    r = model.predict(img, conf=ROAD_CONF, iou=ROAD_IOU, imgsz=ROAD_IMGSZ, max_det=ROAD_MAX_DET, verbose=False)
    out: list = []
    if not r or r[0].boxes is None:
        return out
    names = model.names
    for box in r[0].boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        out.append({
            "label": label,
            "confidence": float(box.conf[0]),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })
    return out


def _dedupe_labelled(dets: list, iou_thresh: float) -> list:
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept: list = []
    for d in dets:
        if not any(k["label"] == d["label"] and _iou(k["box"], d["box"]) > iou_thresh for k in kept):
            kept.append(d)
    return kept


def run_road(model: YOLO | None, img: Image.Image) -> tuple[list, float]:
    if model is None:
        return [], 0.0
    W, H = img.size
    raw = _dedupe_labelled(_predict_road(model, img), iou_thresh=0.5)
    detections = [
        {
            "label": ROAD_LABEL,
            "confidence": round(r["confidence"], 4),
            "box": {"x1": int(r["box"]["x1"]), "y1": int(r["box"]["y1"]),
                    "x2": int(r["box"]["x2"]), "y2": int(r["box"]["y2"])},
        }
        for r in raw
    ]
    return detections, _road_severity(detections, float(W * H))


# ───────────────────────── Waste pipeline (YOLO-E, tiled) ───────────────────────── #
def _waste_pass(model: YOLO, img: Image.Image, x_off: float, y_off: float, imgsz: int) -> list:
    r = model.predict(
        img,
        conf=WASTE_CONF,
        iou=WASTE_IOU,
        imgsz=imgsz,
        max_det=WASTE_MAX_DET,
        device=DEVICE,
        half=USE_HALF,
        verbose=False,
    )
    if not r:
        return []
    res = r[0]
    masks_xy = res.masks.xy if (hasattr(res, "masks") and res.masks is not None) else None
    names = model.names
    out = []
    if res.boxes is None:
        return out
    for i, box in enumerate(res.boxes):
        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()[:4]
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        poly = None
        if masks_xy is not None and i < len(masks_xy) and len(masks_xy[i]) >= 3:
            poly = [(float(px + x_off), float(py + y_off)) for px, py in masks_xy[i]]
        out.append({
            "xyxy": [float(bx1 + x_off), float(by1 + y_off), float(bx2 + x_off), float(by2 + y_off)],
            "conf": float(box.conf[0]),
            "raw": label,
            "polygon": poly,
        })
    return out


def _merge_waste_raws(raws: list) -> list:
    if not raws:
        return []
    for r in raws:
        r["_canon"] = _resolve_category(r["raw"])[0]
    raws = sorted(raws, key=lambda d: -d["conf"])
    kept: list = []
    for d in raws:
        if not any(k["_canon"] == d["_canon"] and _iou_xyxy(d["xyxy"], k["xyxy"]) > WASTE_MERGE_IOU for k in kept):
            kept.append(d)
    return kept


def _waste_tile_predict(model: YOLO, img: Image.Image) -> list:
    # Non-tiled path (CPU-friendly): single 1280 full-image pass.
    if not WASTE_TILED:
        return _merge_waste_raws(_waste_pass(model, img, 0.0, 0.0, WASTE_FULL_IMGSZ))

    W, H = img.size
    stride = max(1, int(WASTE_TILE_SIZE * (1 - WASTE_TILE_OVERLAP)))

    xs = sorted(set(list(range(0, max(1, W - WASTE_TILE_SIZE + 1), stride)) + [max(0, W - WASTE_TILE_SIZE)]))
    ys = sorted(set(list(range(0, max(1, H - WASTE_TILE_SIZE + 1), stride)) + [max(0, H - WASTE_TILE_SIZE)]))

    all_raws: list = []
    for y in ys:
        for x in xs:
            x2, y2 = min(x + WASTE_TILE_SIZE, W), min(y + WASTE_TILE_SIZE, H)
            tile = img.crop((x, y, x2, y2))
            all_raws.extend(_waste_pass(model, tile, float(x), float(y), WASTE_TILE_SIZE))
    all_raws.extend(_waste_pass(model, img, 0.0, 0.0, WASTE_FULL_IMGSZ))
    return _merge_waste_raws(all_raws)


def _maybe_resize(img: Image.Image) -> Image.Image:
    if max(img.size) <= WASTE_MAX_IMAGE_DIM:
        return img
    s = WASTE_MAX_IMAGE_DIM / max(img.size)
    return img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)


def run_waste(model: YOLO | None, img: Image.Image) -> tuple[list, dict, float, float]:
    empty_stats = {"total_detections": 0, "total_coverage_pct": 0.0, "class_counts": {}, "category_counts": {}}
    if model is None:
        return [], empty_stats, 0.0, 0.0

    img = _maybe_resize(img)
    W, H = img.size
    img_area = float(W * H)
    raw = _waste_tile_predict(model, img)

    detections: list = []
    for r in raw:
        x1, y1, x2, y2 = r["xyxy"]
        area_pct = ((x2 - x1) * (y2 - y1)) / img_area * 100.0
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        category, meta = _resolve_category(r["raw"])
        pw = _position_weight(cx, cy, W, H)
        detections.append({
            "label": r["raw"],
            "category": category,
            "confidence": round(r["conf"], 4),
            "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "centroid": [round(cx, 1), round(cy, 1)],
            "area_pct": round(area_pct, 2),
            "severity": _waste_severity(area_pct, r["conf"], meta, pw),
            "environmental_impact": _waste_impact(area_pct, meta),
            "recyclable": meta["recyclable"],
            "decomp_years": meta["decomp_years"],
            # leading underscore → transient, stripped before Mongo insert
            "_polygon": r.get("polygon"),
        })

    if detections:
        total_coverage = round(sum(d["area_pct"] for d in detections), 2)
        avg_sev = sum(d["severity"] for d in detections) / len(detections)
        sev_score = round(min(avg_sev * (1.0 + total_coverage / 100.0), 100.0), 2)
        imp_score = round(sum(d["environmental_impact"] for d in detections) / len(detections), 2)
    else:
        total_coverage, sev_score, imp_score = 0.0, 0.0, 0.0

    stats = {
        "total_detections": len(detections),
        "total_coverage_pct": total_coverage,
        "class_counts": dict(Counter(d["label"] for d in detections)),
        "category_counts": dict(Counter(d["category"] for d in detections)),
    }
    return detections, stats, sev_score, imp_score


# ───────────────────────── Annotation ───────────────────────── #
def _load_font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def annotate(img: Image.Image, waste_dets: list, road_dets: list, waste_sev: float, road_sev: float) -> bytes:
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, _ = base.size
    f_lg = _load_font(max(20, W // 50))
    f_md = _load_font(max(15, W // 70))
    f_sm = _load_font(max(12, W // 90))

    merged = [{**d, "category": d.get("category", "mixed")} for d in waste_dets] + \
             [{**d, "category": "road"} for d in road_dets]

    for d in merged:
        color = CATEGORY_COLORS.get(d["category"], (156, 163, 175))
        poly = d.get("_polygon")
        if poly and len(poly) >= 3:
            draw.polygon(poly, fill=(*color, 80), outline=(*color, 255))
        else:
            b = d["box"]
            draw.rectangle((b["x1"], b["y1"], b["x2"], b["y2"]), outline=(*color, 255), width=3)
        b = d["box"]
        lbl = f"{d['label']} {d['confidence']*100:.0f}%"
        tb = draw.textbbox((b["x1"], b["y1"]), lbl, font=f_sm)
        pad = 4
        draw.rectangle((tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad), fill=(*color, 220))
        draw.text((b["x1"], b["y1"]), lbl, fill=(255, 255, 255, 255), font=f_sm)

    lines = [
        (f"trash {len(waste_dets)} · pothole {len(road_dets)}", f_lg),
        (f"waste severity {waste_sev:.1f}", f_md),
        (f"road severity {road_sev:.1f}", f_md),
    ]
    y = 20
    for text, font in lines:
        draw.text((22, y + 2), text, fill=(0, 0, 0, 200), font=font)
        draw.text((20, y), text, fill=(255, 255, 255, 255), font=font)
        y += font.getbbox(text)[3] + 10

    out = Image.alpha_composite(base, overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
