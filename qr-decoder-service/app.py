import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pyzbar.pyzbar import ZBarSymbol, decode


app = FastAPI(
    title="Pyzbar QR Decoder POC",
    version="1.1.0",
)

QR_API_KEY = os.environ.get("QR_API_KEY", "")
MAX_FILE_SIZE = 15 * 1024 * 1024

# Relative crop coordinates: name, x1, y1, x2, y2.
# These cover common document QR positions while retaining a full-image pass.
CROP_REGIONS = (
    ("bottom_left_35", 0.00, 0.65, 0.35, 1.00),
    ("bottom_left_50", 0.00, 0.50, 0.50, 1.00),
    ("bottom_half", 0.00, 0.50, 1.00, 1.00),
    ("top_left_35", 0.00, 0.00, 0.35, 0.35),
    ("top_right_35", 0.65, 0.00, 1.00, 0.35),
    ("bottom_right_35", 0.65, 0.65, 1.00, 1.00),
)


def validate_api_key(x_api_key):
    if not QR_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="QR_API_KEY is not configured",
        )

    if x_api_key != QR_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


def decode_candidate(candidate):
    values = []

    decoded_results = decode(
        candidate,
        symbols=[ZBarSymbol.QRCODE],
    )

    for item in decoded_results:
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


def add_candidate(candidates, name, image):
    if image is None or image.size == 0:
        return

    candidates.append((name, image))


def add_enhanced_variants(candidates, prefix, image):
    """Create several variants while keeping Pyzbar as the decoder."""
    if image is None or image.size == 0:
        return

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    add_candidate(candidates, f"{prefix}_color", image)
    add_candidate(candidates, f"{prefix}_gray", gray)

    normalized = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )
    add_candidate(candidates, f"{prefix}_normalized", normalized)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)
    add_candidate(candidates, f"{prefix}_clahe", clahe)

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    add_candidate(candidates, f"{prefix}_otsu", otsu)
    add_candidate(
        candidates,
        f"{prefix}_otsu_inverted",
        cv2.bitwise_not(otsu),
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    add_candidate(candidates, f"{prefix}_adaptive", adaptive)
    add_candidate(
        candidates,
        f"{prefix}_adaptive_inverted",
        cv2.bitwise_not(adaptive),
    )

    # Nearest-neighbor keeps QR module edges hard.
    nearest_3x = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_NEAREST,
    )
    add_candidate(candidates, f"{prefix}_nearest_3x", nearest_3x)

    # Cubic enlargement can help when the source is a photographed page.
    cubic_3x = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )
    add_candidate(candidates, f"{prefix}_cubic_3x", cubic_3x)

    # Mild unsharp masking on the cubic enlargement.
    blurred = cv2.GaussianBlur(cubic_3x, (0, 0), 2.0)
    sharpened = cv2.addWeighted(
        cubic_3x,
        1.8,
        blurred,
        -0.8,
        0,
    )
    add_candidate(candidates, f"{prefix}_sharpened_3x", sharpened)


def build_candidates(image):
    height, width = image.shape[:2]
    candidates = []

    # Always try the complete image first.
    add_enhanced_variants(candidates, "full", image)

    # Then try common page regions. This is important when the QR occupies
    # only a small fraction of a full-page image.
    for name, x1, y1, x2, y2 in CROP_REGIONS:
        left = max(0, min(width - 1, int(width * x1)))
        top = max(0, min(height - 1, int(height * y1)))
        right = max(left + 1, min(width, int(width * x2)))
        bottom = max(top + 1, min(height, int(height * y2)))

        crop = image[top:bottom, left:right]
        add_enhanced_variants(candidates, name, crop)

    return candidates


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "decoder": "pyzbar",
        "version": "1.1.0",
    }


@app.post("/decode")
async def decode_qr(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=None),
):
    validate_api_key(x_api_key)

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 15 MB POC limit",
        )

    file_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )
    image = cv2.imdecode(
        file_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not a readable PNG/JPG image. "
                "Render PDF pages to images before calling this endpoint."
            ),
        )

    candidates = build_candidates(image)

    logging.info(
        "Received image width=%s height=%s candidates=%s",
        image.shape[1],
        image.shape[0],
        len(candidates),
    )

    for index, (candidate_name, candidate) in enumerate(candidates):
        try:
            values = decode_candidate(candidate)

            if values:
                logging.info(
                    "QR decoded candidate=%s values=%s",
                    candidate_name,
                    values,
                )
                return {
                    "found": True,
                    "qr_values": values,
                    "decoder": "pyzbar",
                    "candidate": candidate_name,
                    "candidate_index": index,
                    "image_width": image.shape[1],
                    "image_height": image.shape[0],
                }

        except Exception as exc:
            logging.warning(
                "Pyzbar failed candidate=%s error=%s",
                candidate_name,
                exc,
            )

    logging.info("No QR decoded from any candidate")

    return {
        "found": False,
        "qr_values": [],
        "decoder": "pyzbar",
        "candidate": None,
        "candidate_index": None,
        "candidates_tested": len(candidates),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
    }
