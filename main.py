import io
import json
import base64
import re
import math
import os
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# ---------- Tunables ----------
CLS_IMGSZ   = 448
PAD_RATIO   = 0.0
MIN_SIDE    = 24
SKU_FONT_SZ = 14  # draw size for labels


# ============================================================
#  Persian/Latin text helpers
# ============================================================

def _shape_arabic(text: str) -> str:
    """Arabic shaping + bidi for RTL text only."""
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _find_font(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def _load_fonts() -> Tuple[Optional[ImageFont.FreeTypeFont], Optional[ImageFont.FreeTypeFont]]:
    # Allow override via env
    fa_env = os.getenv("PERSIAN_FONT_PATH")
    fa = _find_font([
        fa_env,
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    ])
    la = _find_font([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ])
    f_ar = ImageFont.truetype(fa, size=SKU_FONT_SZ) if fa else None
    f_la = ImageFont.truetype(la, size=SKU_FONT_SZ) if la else None
    return f_ar, f_la


_FONT_AR, _FONT_LAT = _load_fonts()


def _has_arabic(s: str) -> bool:
    """Detect if a string contains Arabic/Persian letters."""
    for ch in s or "":
        o = ord(ch)
        if (0x0600 <= o <= 0x06FF) or (0x0750 <= o <= 0x077F) or \
           (0x08A0 <= o <= 0x08FF) or (0xFB50 <= o <= 0xFDFF) or \
           (0xFE70 <= o <= 0xFEFF):
            return True
    return False


def _draw_mixed_label(
    bgr_img: np.ndarray, x: int, y: int,
    rtl_text: str = "", ltr_text: str = "",
    color=(36, 255, 12), bg=(0, 0, 0), gap: int = 8
) -> np.ndarray:
    """
    Draw RTL (Persian) segment with Arabic font + LTR (Latin) segment with Latin font.
    (x,y) is baseline-left like cv2.putText.
    """
    pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    f_ar = _FONT_AR or _FONT_LAT
    f_la = _FONT_LAT or _FONT_AR

    if not (f_ar and f_la):
        # Fallback: everything with cv2 (Persian may appear as '?')
        txt = (rtl_text + " " + ltr_text).strip()
        cv2.putText(bgr_img, txt, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return bgr_img

    rtl_shaped = _shape_arabic(rtl_text) if rtl_text else ""

    pad_x, pad_y = 4, 3
    rtl_w = rtl_h = ltr_w = ltr_h = 0
    if rtl_shaped:
        rb = draw.textbbox((0, 0), rtl_shaped, font=f_ar)
        rtl_w, rtl_h = rb[2] - rb[0], rb[3] - rb[1]
    if ltr_text:
        lb = draw.textbbox((0, 0), ltr_text, font=f_la)
        ltr_w, ltr_h = lb[2] - lb[0], lb[3] - lb[1]

    sep = (gap if rtl_shaped and ltr_text else 0)
    total_w = rtl_w + sep + ltr_w + 2 * pad_x
    total_h = max(rtl_h, ltr_h, 1) + 2 * pad_y

    top = y - total_h + 2
    left = x
    draw.rectangle([left, top, left + total_w, top + total_h], fill=bg)

    cx = left + pad_x
    if rtl_shaped:
        draw.text((cx, top + pad_y), rtl_shaped, font=f_ar, fill=tuple(color))
        cx += rtl_w + sep
    if ltr_text:
        draw.text((cx, top + pad_y), ltr_text, font=f_la, fill=tuple(color))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ============================================================
#  General helpers
# ============================================================

def _norm_class_key(s: str) -> str:
    # Unicode-normalize (handles Persian), remove separators for robust matching
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u200c", " ")
    s = " ".join(s.split())
    s = re.sub(r"[\s_\-]+", "", s)
    return s.lower()


def _letterbox_square_bgr(img_bgr: np.ndarray, out_size: int, pad_val: int = 114) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = out_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_size, out_size, 3), pad_val, dtype=np.uint8)
    ys = (out_size - nh) // 2
    xs = (out_size - nw) // 2
    canvas[ys:ys + nh, xs:xs + nw] = resized
    return canvas


def _pad_clip(x1, y1, x2, y2, W, H, pad=PAD_RATIO):
    w = x2 - x1
    h = y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    w2, h2 = w * (1.0 + pad), h * (1.0 + pad)
    xx1, yy1 = int(max(0, math.floor(cx - w2 / 2))), int(max(0, math.floor(cy - h2 / 2)))
    xx2, yy2 = int(min(W - 1, math.ceil(cx + w2 / 2))), int(min(H - 1, math.ceil(cy + h2 / 2)))
    return xx1, yy1, xx2, yy2


def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def pil_to_base64_jpeg(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ============================================================
#  Loading models
# ============================================================

def load_models_and_maps(
    det_path: str,
    cls_path: str,
    class_to_idxs_path: Optional[str] = None,
    sku_map_path: Optional[str] = None,
) -> Dict[str, Any]:
    det = YOLO(det_path)
    cls = YOLO(cls_path)
    det.fuse()
    cls.fuse()

    class_to_idxs_norm = {}
    if class_to_idxs_path and Path(class_to_idxs_path).exists():
        raw = json.loads(Path(class_to_idxs_path).read_text(encoding="utf-8"))
        class_to_idxs_norm = {_norm_class_key(k): v for k, v in raw.items()}

    sku_map = {}
    if sku_map_path and Path(sku_map_path).exists():
        sku_map = json.loads(Path(sku_map_path).read_text(encoding="utf-8"))
        # expected keys: "idx_to_sku_pair": {"0": {"class":..., "flavor":...}, ...}

    return {
        "det": det,
        "cls": cls,
        "det_names": det.names,   # id->class
        "cls_names": cls.names,   # id->folder_name like "0004__Class__Flavor"
        "class_to_idxs_norm": class_to_idxs_norm,
        "sku_map": sku_map,
    }


# ============================================================
#  Core 2-layer inference for one bundle
# ============================================================

def _detect_and_classify_items(
    bundle: Dict[str, Any],
    im_bgr: np.ndarray,
    det_conf: float,
    det_iou: float,
    sku_accept_threshold: float,
    do_tta: bool,
) -> List[Dict[str, Any]]:
    """
    Core 2-layer pipeline for ONE bundle.
    Runs detector + classifier and returns a list of items with boxes and labels,
    but does NOT draw on the image.
    """
    det, cls = bundle["det"], bundle["cls"]
    det_names = bundle["det_names"]
    cls_names = bundle["cls_names"]
    class_to_idxs_norm = bundle.get("class_to_idxs_norm", {})
    sku_map = bundle.get("sku_map", {})

    H, W = im_bgr.shape[:2]

    # --- Detector ---
    det_res = det.predict(
        source=im_bgr,      # BGR np array is fine
        imgsz=640,
        conf=det_conf,
        iou=det_iou,
        max_det=300,
        save=False,
        verbose=False
    )[0]

    meta = []
    for b in det_res.boxes or []:
        x1, y1, x2, y2 = map(float, b.xyxy.cpu().numpy().ravel().tolist())
        cls_id = int(b.cls.item())
        det_conf_box = float(b.conf.item())
        c_name = det_names.get(cls_id, str(cls_id))
        xx1, yy1, xx2, yy2 = _pad_clip(x1, y1, x2, y2, W, H, PAD_RATIO)
        if min(xx2 - xx1, yy2 - yy1) < MIN_SIDE:
            continue
        crop = im_bgr[yy1:yy2, xx1:xx2]
        crop_sq = _letterbox_square_bgr(crop, CLS_IMGSZ, 114)
        meta.append({
            "det_class": c_name,
            "det_conf": det_conf_box,
            "xyxy": (xx1, yy1, xx2, yy2),
            "crop_sq": crop_sq,
        })

    if not meta:
        return []

    # --- Classifier (with optional TTA) ---
    crops = [m["crop_sq"] for m in meta]

    def _predict_logits(images_bgr: List[np.ndarray]) -> List[np.ndarray]:
        preds = cls.predict(source=images_bgr, imgsz=CLS_IMGSZ, save=False, verbose=False)
        return [p.probs.data.float().cpu().numpy() for p in preds]

    logits = _predict_logits(crops)
    if do_tta:
        flips = [cv2.flip(c, 1) for c in crops]
        logits2 = _predict_logits(flips)
        logits = [(a + b) / 2.0 for a, b in zip(logits, logits2)]

    items: List[Dict[str, Any]] = []

    for logit, m in zip(logits, meta):
        c_name = m["det_class"]
        det_conf_box = m["det_conf"]
        xx1, yy1, xx2, yy2 = m["xyxy"]

        allowed = class_to_idxs_norm.get(_norm_class_key(c_name), [])
        probs = logit

        if allowed:
            mask = np.full_like(probs, -np.inf, dtype=np.float32)
            mask[allowed] = 0.0
            masked = probs + mask
            mmax = float(np.max(masked))
            ex = np.exp(masked - mmax)
            p = ex / (np.sum(ex) + 1e-9)
        else:
            mmax = float(np.max(probs))
            ex = np.exp(probs - mmax)
            p = ex / (np.sum(ex) + 1e-9)

        top_idx = int(np.argmax(p))
        top_prob = float(p[top_idx])
        folder = cls_names[top_idx]  # e.g. "0004__Class__Flavor"

        # Decode SKU class + flavor from sku_map if available
        sk_class = c_name
        sk_flavor = ""
        try:
            prefix = int(folder.split("__", 1)[0])
            meta_info = sku_map.get("idx_to_sku_pair", {}).get(str(prefix))
            if isinstance(meta_info, dict):
                sk_class = meta_info.get("class", c_name)
                sk_flavor = meta_info.get("flavor", "")
        except Exception:
            pass

        if top_prob < sku_accept_threshold:
            sk_flavor = "unknown"

        # Build text segments (for drawing)
        rtl_parts, ltr_parts = [], []
        if _has_arabic(sk_class):
            rtl_parts.append(sk_class)
        elif sk_class:
            ltr_parts.append(sk_class)

        flavor_for_draw = "نامشخص" if sk_flavor == "unknown" else sk_flavor
        if _has_arabic(flavor_for_draw):
            rtl_parts.append(flavor_for_draw)
        elif flavor_for_draw:
            ltr_parts.append(flavor_for_draw)

        rtl_label = " | ".join([p for p in rtl_parts if p])
        ltr_head  = " | ".join([p for p in ltr_parts if p])
        ltr_tail  = (ltr_head + "  " if ltr_head else "") + f"det:{det_conf_box:.2f} cls:{top_prob:.2f}"

        items.append({
            "category": c_name,
            "sku": sk_flavor,
            "det_conf": float(det_conf_box),
            "sku_prob": float(top_prob),
            "x1": float(xx1),
            "y1": float(yy1),
            "x2": float(xx2),
            "y2": float(yy2),
            "rtl_label": rtl_label,
            "ltr_tail": ltr_tail,
        })

    return items


# ============================================================
#  Nuclio integration
# ============================================================

def init_context(context):
    """Called once when the function container starts."""
    context.logger.info("Init b2shelf two-layer function...")

    base = "/opt/nuclio/models"   # where we’ll mount models from the host

    context.user_data.drink_bundle = load_models_and_maps(
        det_path=f"{base}/drink/FirstModel.pt",
        cls_path=f"{base}/drink/SecondModel.pt",
        class_to_idxs_path=f"{base}/drink/class_to_classifier_idxs_norm.json",
        sku_map_path=f"{base}/drink/sku_mapping.json",
    )

    context.user_data.bakery_bundle = load_models_and_maps(
        det_path=f"{base}/bakery/FirstModel.pt",
        cls_path=f"{base}/bakery/SecondModel.pt",
        class_to_idxs_path=f"{base}/bakery/class_to_classifier_idxs_norm.json",
        sku_map_path=f"{base}/bakery/sku_mapping.json",
    )

    context.logger.info("Models loaded successfully")


def handler(context, event):
    """
    Nuclio handler used by CVAT automatic annotation.

    Input (from CVAT):
    {
        "image": "<base64>",
        "threshold": 0.6
    }

    Output:
    [
      {
        "confidence": "0.92",
        "label": "Sunich_Pak_B",
        "points": [x1, y1, x2, y2],
        "type": "rectangle",
        "attributes": [
          {"name": "Flavor", "value": "پرتقال"}
        ]
      },
      ...
    ]
    """
    try:
        body = event.body
        if isinstance(body, (bytes, str)):
            body = json.loads(body)
        data = body

        img_b64 = data["image"]
        det_conf = float(data.get("threshold", 0.6))
        det_iou = float(data.get("iou", 0.3))
        sku_threshold = float(data.get("sku_threshold", 0.05))
        do_tta = bool(data.get("tta", True))

        img_bytes = base64.b64decode(img_b64)
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        im_bgr = _pil_to_bgr(image_pil)

        items_drink = _detect_and_classify_items(
            context.user_data.drink_bundle,
            im_bgr,
            det_conf=det_conf,
            det_iou=det_iou,
            sku_accept_threshold=sku_threshold,
            do_tta=do_tta,
        )

        items_bakery = _detect_and_classify_items(
            context.user_data.bakery_bundle,
            im_bgr,
            det_conf=det_conf,
            det_iou=det_iou,
            sku_accept_threshold=sku_threshold,
            do_tta=do_tta,
        )

        all_items = items_drink + items_bakery

        results = []
        for it in all_items:
            x1, y1, x2, y2 = it["x1"], it["y1"], it["x2"], it["y2"]

            category = it["category"] or "Unknown"
            flavor = it["sku"] or "unknown"

            # Map unknown to CVAT's default flavor option
            if flavor == "unknown":
                flavor = "default"

            results.append({
                "confidence": f"{float(it['det_conf']):.3f}",
                "label": category,
                "points": [float(x1), float(y1), float(x2), float(y2)],
                "type": "rectangle",
                "attributes": [
                    {
                        "name": "Flavor",
                        "value": flavor,
                    }
                ],
            })

        return context.Response(
            body=json.dumps(results),
            content_type="application/json",
            status_code=200,
        )

    except Exception as exc:
        context.logger.error(f"Handler error: {exc!r}")
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            content_type="application/json",
            status_code=500,
        )
