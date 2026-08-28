import io
import os
import logging

import cv2
import numpy as np

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode, ZBarSymbol


app = FastAPI(
    title="Pyzbar QR Decoder POC",
    version="1.0.0",
)

QR_API_KEY = os.environ.get("QR_API_KEY", "")

MAX_FILE_SIZE = 15 * 1024 * 1024


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


def pyzbar_decode(candidate):
    results = decode(
        candidate,
        symbols=[ZBarSymbol.QRCODE],
    )

    values = []

    for result in results:
        try:
            value = result.data.decode(
                "utf-8",
                errors="strict",
            ).strip()
        except UnicodeDecodeError:
            value = result.data.decode(
                "latin-1",
                errors="replace",
            ).strip()

        if value and value not in values:
            values.append(value)

    return values


def build_image_variants(image_bytes):
    file_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        file_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "The uploaded file is not a readable image"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    variants = [
        image,
        gray,
    ]

    # Contrast normalization.
    normalized = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )
    variants.append(normalized)

    # CLAHE can improve local contrast in uneven scans.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    clahe_image = clahe.apply(gray)
    variants.append(clahe_image)

    # Otsu black-and-white version.
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    variants.append(otsu)

    # Adaptive threshold version.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    variants.append(adaptive)

    # Enlarged versions help when QR modules are very small.
    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )
    variants.append(enlarged)

    enlarged_clahe = cv2.resize(
        clahe_image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )
    variants.append(enlarged_clahe)

    # Try inverted images for reversed or badly scanned contrast.
    variants.append(cv2.bitwise_not(otsu))
    variants.append(cv2.bitwise_not(adaptive))

    return variants


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "decoder": "pyzbar",
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

    try:
        variants = build_image_variants(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    for index, candidate in enumerate(variants):
        try:
            values = pyzbar_decode(candidate)

            if values:
                return {
                    "found": True,
                    "qr_values": values,
                    "decoder": "pyzbar",
                    "variant_index": index,
                }

        except Exception as exc:
            logging.warning(
                "Pyzbar failed on variant %s: %s",
                index,
                exc,
            )

    return {
        "found": False,
        "qr_values": [],
        "decoder": "pyzbar",
        "variant_index": None,
    }
