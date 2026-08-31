import logging
import os
import time

import cv2
import numpy as np
import zxingcpp
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from pyzbar.pyzbar import ZBarSymbol, decode

APP_VERSION = "1.3.1"
MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_EXHAUSTIVE_CANDIDATES = 60
QR_API_KEY = os.environ.get("QR_API_KEY", "")

app = FastAPI(
    title="Multi-format Barcode Decoder",
    version=APP_VERSION,
)

# Relative page regions: name, x1, y1, x2, y2.
CROP_REGIONS = (
    ("bottom_left_25", 0.00, 0.72, 0.25, 1.00),
    ("bottom_left_35", 0.00, 0.62, 0.35, 1.00),
    ("bottom_left_50", 0.00, 0.48, 0.50, 1.00),
    ("bottom_half", 0.00, 0.50, 1.00, 1.00),
    ("top_left_35", 0.00, 0.00, 0.35, 0.35),
    ("top_right_35", 0.65, 0.00, 1.00, 0.35),
    ("bottom_right_35", 0.65, 0.65, 1.00, 1.00),
)

FAST_REGION_NAMES = {
    "full",
    "bottom_left_25",
    "bottom_left_35",
    "bottom_left_50",
    "bottom_half",
}


def validate_api_key(value):
    if not QR_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="QR_API_KEY is not configured",
        )
    if value != QR_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def normalize_format(barcode_format):
    value = str(barcode_format)
    return value.split(".")[-1] if "." in value else value


def zxing_decode(image):
    try:
        results = zxingcpp.read_barcodes(
            image,
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
        decoded = []
        for barcode in results:
            value = (barcode.text or "").strip()
            if value:
                decoded.append(
                    {
                        "value": value,
                        "format": normalize_format(barcode.format),
                    }
                )
        return decoded
    except Exception as exc:
        logging.exception("ZXing decode failed: %s", exc)
        return []


def pyzbar_decode(image):
    try:
        results = decode(image, symbols=[ZBarSymbol.QRCODE])
    except Exception as exc:
        logging.warning("Pyzbar decode failed: %s", exc)
        return []

    values = []
    for item in results:
        try:
            value = item.data.decode("utf-8").strip()
        except UnicodeDecodeError:
            value = item.data.decode(
                "latin-1",
                errors="replace",
            ).strip()
        if value and value not in values:
            values.append(value)
    return values


def add_quiet_zone(image, border_ratio=0.10):
    height, width = image.shape[:2]
    border = max(12, int(max(height, width) * border_ratio))
    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(
        image,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )


def region_images(image):
    height, width = image.shape[:2]
    yield "full", image

    for name, x1, y1, x2, y2 in CROP_REGIONS:
        left = max(0, min(width - 1, int(width * x1)))
        top = max(0, min(height - 1, int(height * y1)))
        right = max(left + 1, min(width, int(width * x2)))
        bottom = max(top + 1, min(height, int(height * y2)))
        crop = image[top:bottom, left:right]
        if crop.size:
            yield name, crop


def fast_candidate_images(image):
    """
    Lightweight boundary scan. This path is used for every PDF page.
    It deliberately avoids the expensive exhaustive preprocessing that can
    block later page requests after a page without a barcode.
    """
    for region_name, region in region_images(image):
        if region_name not in FAST_REGION_NAMES:
            continue

        yield f"{region_name}_color", region

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        yield f"{region_name}_gray", gray

        # A single bounded enlargement helps small symbols without creating
        # long-running page requests.
        if region_name != "full":
            enlarged = cv2.resize(
                gray,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC,
            )
            yield (
                f"{region_name}_cubic_3x",
                add_quiet_zone(enlarged),
            )


def exhaustive_candidate_images(image):
    """Full fallback for individual documents or difficult barcode images."""
    for region_name, region in region_images(image):
        yield f"{region_name}_color", region

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        yield f"{region_name}_gray", gray

        normalized = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )
        yield f"{region_name}_normalized", normalized

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8),
        ).apply(gray)
        yield f"{region_name}_clahe", clahe

        for scale in (3, 5):
            nearest = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
            yield (
                f"{region_name}_nearest_{scale}x",
                add_quiet_zone(nearest),
            )

            cubic = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            yield (
                f"{region_name}_cubic_{scale}x",
                add_quiet_zone(cubic),
            )

            _, otsu = cv2.threshold(
                cubic,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            yield (
                f"{region_name}_otsu_{scale}x",
                add_quiet_zone(otsu),
            )


def order_points(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def perspective_warp(image, points, output_size=900):
    source = order_points(points)
    margin = int(output_size * 0.12)
    destination = np.array(
        [
            [margin, margin],
            [output_size - margin, margin],
            [output_size - margin, output_size - margin],
            [margin, output_size - margin],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def opencv_qr_decode_and_localize(image):
    detector = cv2.QRCodeDetector()
    try:
        detector.setEpsX(0.35)
        detector.setEpsY(0.35)
    except Exception:
        pass

    try:
        value, points, straight = detector.detectAndDecode(image)
        if value:
            return value.strip(), None, True
        if straight is not None and getattr(straight, "size", 0) > 0:
            return None, straight, True
        if points is not None and np.asarray(points).size >= 8:
            return None, perspective_warp(image, points), True
    except Exception as exc:
        logging.debug("OpenCV detectAndDecode failed: %s", exc)

    return None, None, False


def success_response(
    values,
    decoder,
    barcode_type,
    candidate,
    image_width,
    image_height,
    candidates_tested,
    processing_seconds,
    mode,
):
    return {
        "found": True,
        "qr_values": values,
        "decoder": decoder,
        "barcode_type": barcode_type,
        "candidate": candidate,
        "candidates_tested": candidates_tested,
        "image_width": image_width,
        "image_height": image_height,
        "processing_seconds": round(processing_seconds, 3),
        "mode": mode,
        "version": APP_VERSION,
    }


@app.get("/")
def root():
    return {
        "service": "multi-format-barcode-decoder",
        "status": "healthy",
        "version": APP_VERSION,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "decoder": "zxing+pyzbar+opencv",
        "version": APP_VERSION,
    }


@app.post("/decode")
async def decode_barcode(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=None),
    mode: str = Query(default="exhaustive"),
):
    validate_api_key(x_api_key)
    started = time.monotonic()
    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 15 MB limit",
        )

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload a readable PNG/JPG image. "
                "Render PDF pages before calling /decode."
            ),
        )

    mode = mode.strip().lower()
    if mode not in {"fast", "exhaustive"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be fast or exhaustive",
        )

    image_height, image_width = image.shape[:2]
    candidates_tested = 0
    generator = (
        fast_candidate_images(image)
        if mode == "fast"
        else exhaustive_candidate_images(image)
    )

    for candidate_name, candidate in generator:
        if (
            mode == "exhaustive"
            and candidates_tested >= MAX_EXHAUSTIVE_CANDIDATES
        ):
            break

        candidates_tested += 1

        zxing_results = zxing_decode(candidate)
        if zxing_results:
            return success_response(
                values=[item["value"] for item in zxing_results],
                decoder="zxing",
                barcode_type=zxing_results[0]["format"],
                candidate=candidate_name,
                image_width=image_width,
                image_height=image_height,
                candidates_tested=candidates_tested,
                processing_seconds=time.monotonic() - started,
                mode=mode,
            )

        pyzbar_values = pyzbar_decode(candidate)
        if pyzbar_values:
            return success_response(
                values=pyzbar_values,
                decoder="pyzbar",
                barcode_type="QRCode",
                candidate=candidate_name,
                image_width=image_width,
                image_height=image_height,
                candidates_tested=candidates_tested,
                processing_seconds=time.monotonic() - started,
                mode=mode,
            )

        # OpenCV is skipped in fast mode because ZXing handles QR and Data
        # Matrix and the splitter only needs a quick boundary decision.
        if mode == "exhaustive":
            opencv_value, rectified, _ = (
                opencv_qr_decode_and_localize(candidate)
            )
            if opencv_value:
                return success_response(
                    values=[opencv_value],
                    decoder="opencv",
                    barcode_type="QRCode",
                    candidate=candidate_name,
                    image_width=image_width,
                    image_height=image_height,
                    candidates_tested=candidates_tested,
                    processing_seconds=time.monotonic() - started,
                    mode=mode,
                )

            if rectified is not None:
                candidates_tested += 1
                zxing_rectified = zxing_decode(rectified)
                if zxing_rectified:
                    return success_response(
                        values=[
                            item["value"]
                            for item in zxing_rectified
                        ],
                        decoder="zxing_after_opencv_localization",
                        barcode_type=zxing_rectified[0]["format"],
                        candidate=f"{candidate_name}_rectified",
                        image_width=image_width,
                        image_height=image_height,
                        candidates_tested=candidates_tested,
                        processing_seconds=time.monotonic() - started,
                        mode=mode,
                    )

    return {
        "found": False,
        "qr_values": [],
        "decoder": "zxing+pyzbar+opencv",
        "barcode_type": None,
        "candidate": None,
        "candidates_tested": candidates_tested,
        "image_width": image_width,
        "image_height": image_height,
        "processing_seconds": round(
            time.monotonic() - started,
            3,
        ),
        "mode": mode,
        "version": APP_VERSION,
        "note": "No supported barcode payload was recovered.",
    }
