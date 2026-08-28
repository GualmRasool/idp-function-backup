import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pyzbar.pyzbar import ZBarSymbol, decode

APP_VERSION = "1.2.0"
app = FastAPI(title="Pyzbar QR Decoder POC", version=APP_VERSION)

QR_API_KEY = os.environ.get("QR_API_KEY", "")
MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_CANDIDATES = 140

# name, x1, y1, x2, y2 using relative coordinates
CROP_REGIONS = (
    ("bottom_left_25", 0.00, 0.72, 0.25, 1.00),
    ("bottom_left_35", 0.00, 0.62, 0.35, 1.00),
    ("bottom_left_50", 0.00, 0.48, 0.50, 1.00),
    ("bottom_half", 0.00, 0.50, 1.00, 1.00),
    ("top_left_35", 0.00, 0.00, 0.35, 0.35),
    ("top_right_35", 0.65, 0.00, 1.00, 0.35),
    ("bottom_right_35", 0.65, 0.65, 1.00, 1.00),
)


def validate_api_key(value):
    if not QR_API_KEY:
        raise HTTPException(status_code=500, detail="QR_API_KEY is not configured")
    if value != QR_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def unique_values(results):
    values = []
    for item in results:
        try:
            value = item.data.decode("utf-8").strip()
        except UnicodeDecodeError:
            value = item.data.decode("latin-1", errors="replace").strip()
        if value and value not in values:
            values.append(value)
    return values


def pyzbar_decode(image):
    return unique_values(decode(image, symbols=[ZBarSymbol.QRCODE]))


def add_candidate(items, name, image):
    if image is None or image.size == 0 or len(items) >= MAX_CANDIDATES:
        return
    items.append((name, image))


def add_quiet_zone(image, border_ratio=0.12):
    h, w = image.shape[:2]
    border = max(12, int(max(h, w) * border_ratio))
    return cv2.copyMakeBorder(
        image, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255,
    )


def enhanced_variants(prefix, image):
    items = []
    if image is None or image.size == 0:
        return items

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    add_candidate(items, f"{prefix}_color", image)
    add_candidate(items, f"{prefix}_gray", gray)

    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    add_candidate(items, f"{prefix}_normalized", normalized)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    add_candidate(items, f"{prefix}_clahe", clahe)

    for scale in (3, 5, 8):
        nearest = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        cubic = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        add_candidate(items, f"{prefix}_nearest_{scale}x", add_quiet_zone(nearest))
        add_candidate(items, f"{prefix}_cubic_{scale}x", add_quiet_zone(cubic))

        blurred = cv2.GaussianBlur(cubic, (0, 0), 1.5)
        sharpened = cv2.addWeighted(cubic, 1.8, blurred, -0.8, 0)
        add_candidate(items, f"{prefix}_sharp_{scale}x", add_quiet_zone(sharpened))

        _, otsu = cv2.threshold(
            cubic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        add_candidate(items, f"{prefix}_otsu_{scale}x", add_quiet_zone(otsu))

        adaptive = cv2.adaptiveThreshold(
            cubic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5,
        )
        add_candidate(items, f"{prefix}_adaptive_{scale}x", add_quiet_zone(adaptive))

    return items


def order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]   # top-left
    ordered[2] = pts[np.argmax(sums)]   # bottom-right
    ordered[1] = pts[np.argmin(diffs)] # top-right
    ordered[3] = pts[np.argmax(diffs)] # bottom-left
    return ordered


def perspective_warp(image, points, output_size=900, margin_ratio=0.15):
    src = order_points(points)
    margin = int(output_size * margin_ratio)
    dst = np.array([
        [margin, margin],
        [output_size - margin, margin],
        [output_size - margin, output_size - margin],
        [margin, output_size - margin],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image, matrix, (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def opencv_localize(image, prefix):
    """Return direct decode values, localized perspective candidates, and count."""
    detector = cv2.QRCodeDetector()
    localized = []
    detected_count = 0

    # Relax detector tolerances slightly for photographed/low-resolution pages.
    try:
        detector.setEpsX(0.35)
        detector.setEpsY(0.35)
    except Exception:
        pass

    direct_methods = [
        ("standard", detector.detectAndDecode),
    ]
    if hasattr(detector, "detectAndDecodeCurved"):
        direct_methods.append(("curved", detector.detectAndDecodeCurved))

    for method_name, method in direct_methods:
        try:
            data, points, straight = method(image)
            if data:
                return [data.strip()], localized, detected_count + 1
            if points is not None and np.asarray(points).size >= 8:
                detected_count += 1
                warp = perspective_warp(image, points, 1000)
                localized.extend(enhanced_variants(f"{prefix}_{method_name}_warp", warp))
            if straight is not None and getattr(straight, "size", 0) > 0:
                localized.extend(enhanced_variants(f"{prefix}_{method_name}_straight", straight))
        except Exception as exc:
            logging.debug("OpenCV %s failed prefix=%s error=%s", method_name, prefix, exc)

    # Detection-only path, then perspective correction and decoding.
    try:
        found, points = detector.detect(image)
        if found and points is not None:
            detected_count += 1
            warp = perspective_warp(image, points, 1200)
            localized.extend(enhanced_variants(f"{prefix}_detect_warp", warp))
    except Exception as exc:
        logging.debug("OpenCV detect failed prefix=%s error=%s", prefix, exc)

    return [], localized[:40], detected_count


def region_images(image):
    h, w = image.shape[:2]
    regions = [("full", image)]
    for name, x1, y1, x2, y2 in CROP_REGIONS:
        left, top = int(w * x1), int(h * y1)
        right, bottom = int(w * x2), int(h * y2)
        crop = image[top:bottom, left:right]
        if crop.size:
            regions.append((name, crop))
    return regions


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "decoder": "pyzbar+opencv-localizer",
        "version": APP_VERSION,
    }


@app.post("/decode")
async def decode_qr(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=None),
):
    validate_api_key(x_api_key)
    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 15 MB")

    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a readable PNG/JPG image; render PDFs before calling /decode",
        )

    h, w = image.shape[:2]
    candidates = []
    localized_candidates = []
    opencv_detected_count = 0

    # First use OpenCV for direct decoding and QR quadrangle localization.
    for region_name, region in region_images(image):
        values, localized, count = opencv_localize(region, region_name)
        opencv_detected_count += count
        if values:
            return {
                "found": True,
                "qr_values": values,
                "decoder": "opencv",
                "candidate": region_name,
                "image_width": w,
                "image_height": h,
                "opencv_detected_count": opencv_detected_count,
            }
        localized_candidates.extend(localized)
        candidates.extend(enhanced_variants(region_name, region))

    # Prioritize perspective-corrected candidates, then general variants.
    all_candidates = (localized_candidates + candidates)[:MAX_CANDIDATES]
    for index, (candidate_name, candidate) in enumerate(all_candidates):
        try:
            values = pyzbar_decode(candidate)
            if values:
                return {
                    "found": True,
                    "qr_values": values,
                    "decoder": "pyzbar_after_opencv_localization",
                    "candidate": candidate_name,
                    "candidate_index": index,
                    "candidates_tested": index + 1,
                    "image_width": w,
                    "image_height": h,
                    "opencv_detected_count": opencv_detected_count,
                    "localized_candidates": len(localized_candidates),
                }
        except Exception as exc:
            logging.warning(
                "Candidate decode failed name=%s error=%s", candidate_name, exc
            )

    return {
        "found": False,
        "qr_values": [],
        "decoder": "pyzbar+opencv-localizer",
        "candidate": None,
        "candidate_index": None,
        "candidates_tested": len(all_candidates),
        "image_width": w,
        "image_height": h,
        "opencv_detected_count": opencv_detected_count,
        "localized_candidates": len(localized_candidates),
        "note": (
            "No decoder recovered a payload. If opencv_detected_count is 0, "
            "the QR quadrangle was not located in the supplied pixels."
        ),
    }
