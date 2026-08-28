import logging
import os

import cv2
import numpy as np
import zxingcpp
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pyzbar.pyzbar import ZBarSymbol, decode


APP_VERSION = "1.3.0"
MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_CANDIDATES = 60
QR_API_KEY = os.environ.get("QR_API_KEY", "")

app = FastAPI(
    title="Multi-format Barcode Decoder",
    version=APP_VERSION,
)

# Relative page regions: name, x1, y1, x2, y2.
# The lower-left regions target the known COA layout while the remaining
# regions support other document layouts.
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
        raise HTTPException(
            status_code=500,
            detail="QR_API_KEY is not configured",
        )

    if value != QR_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


def normalize_format(barcode_format):
    value = str(barcode_format)

    # zxing-cpp versions may stringify enum values differently.
    if "." in value:
        value = value.split(".")[-1]

    return value


def zxing_decode(image):
    """Read all supported formats with ZXing-C++."""
    try:
        logging.info(
            "Running ZXing image_width=%s image_height=%s",
            image.shape[1],
            image.shape[0],
        )

        results = zxingcpp.read_barcodes(
            image,
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )

        logging.info("ZXing results count=%s", len(results))

        decoded_items = []

        for barcode in results:
            value = (barcode.text or "").strip()

            if not value:
                continue

            item = {
                "value": value,
                "format": normalize_format(barcode.format),
            }

            logging.info(
                "ZXing decoded format=%s value=%s",
                item["format"],
                item["value"],
            )

            decoded_items.append(item)

        return decoded_items

    except Exception as exc:
        logging.exception("ZXing decode failed: %s", exc)
        return []


def pyzbar_decode(image):
    """Decode standard QR codes with Pyzbar/ZBar."""
    try:
        results = decode(
            image,
            symbols=[ZBarSymbol.QRCODE],
        )
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


def candidate_images(region_name, image):
    """
    Yield candidates one at a time to avoid retaining large image copies.
    This keeps memory controlled while trying multi-format decoding.
    """
    yield f"{region_name}_color", image

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
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

    # Keep scaling bounded. The previous 8x/140-candidate approach consumed
    # excessive memory. Candidates are now generated and released sequentially.
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

        del nearest
        del cubic
        del otsu


def order_points(points):
    points = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(4, 2)

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

    matrix = cv2.getPerspectiveTransform(
        source,
        destination,
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def opencv_qr_decode_and_localize(image):
    """Try OpenCV QR decoding and return a rectified candidate if located."""
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

    try:
        found, points = detector.detect(image)

        if found and points is not None:
            return None, perspective_warp(image, points), True

    except Exception as exc:
        logging.debug("OpenCV detect failed: %s", exc)

    return None, None, False


def success_response(
    values,
    decoder,
    barcode_type,
    candidate,
    image_width,
    image_height,
    candidates_tested,
    opencv_detected_count,
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
        "opencv_detected_count": opencv_detected_count,
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
):
    validate_api_key(x_api_key)

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 15 MB limit",
        )

    image_array = np.frombuffer(
        raw,
        dtype=np.uint8,
    )
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload a readable PNG/JPG image. "
                "Render PDF pages to images before calling /decode."
            ),
        )

    image_height, image_width = image.shape[:2]
    candidates_tested = 0
    opencv_detected_count = 0
    formats_seen = []

    logging.info(
        "Decode started version=%s width=%s height=%s bytes=%s",
        APP_VERSION,
        image_width,
        image_height,
        len(raw),
    )

    for region_name, region in region_images(image):
        for candidate_name, candidate in candidate_images(
            region_name,
            region,
        ):
            if candidates_tested >= MAX_CANDIDATES:
                break

            candidates_tested += 1

            # ZXing-C++ is the primary multi-format decoder. It can identify
            # QR Code, Data Matrix, PDF417, Aztec and other supported formats.
            zxing_results = zxing_decode(candidate)

            if zxing_results:
                values = [item["value"] for item in zxing_results]
                barcode_type = zxing_results[0]["format"]

                return success_response(
                    values=values,
                    decoder="zxing",
                    barcode_type=barcode_type,
                    candidate=candidate_name,
                    image_width=image_width,
                    image_height=image_height,
                    candidates_tested=candidates_tested,
                    opencv_detected_count=opencv_detected_count,
                )

            # Retain Pyzbar as a proven QR fallback for existing good samples.
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
                    opencv_detected_count=opencv_detected_count,
                )

            # OpenCV remains a QR-specific fallback and localization helper.
            opencv_value, rectified, detected = (
                opencv_qr_decode_and_localize(candidate)
            )

            if detected:
                opencv_detected_count += 1

            if opencv_value:
                return success_response(
                    values=[opencv_value],
                    decoder="opencv",
                    barcode_type="QRCode",
                    candidate=candidate_name,
                    image_width=image_width,
                    image_height=image_height,
                    candidates_tested=candidates_tested,
                    opencv_detected_count=opencv_detected_count,
                )

            if rectified is not None:
                rectified_name = f"{candidate_name}_rectified"
                candidates_tested += 1

                zxing_rectified = zxing_decode(rectified)

                if zxing_rectified:
                    values = [
                        item["value"]
                        for item in zxing_rectified
                    ]
                    barcode_type = zxing_rectified[0]["format"]

                    return success_response(
                        values=values,
                        decoder="zxing_after_opencv_localization",
                        barcode_type=barcode_type,
                        candidate=rectified_name,
                        image_width=image_width,
                        image_height=image_height,
                        candidates_tested=candidates_tested,
                        opencv_detected_count=opencv_detected_count,
                    )

                pyzbar_rectified = pyzbar_decode(rectified)

                if pyzbar_rectified:
                    return success_response(
                        values=pyzbar_rectified,
                        decoder="pyzbar_after_opencv_localization",
                        barcode_type="QRCode",
                        candidate=rectified_name,
                        image_width=image_width,
                        image_height=image_height,
                        candidates_tested=candidates_tested,
                        opencv_detected_count=opencv_detected_count,
                    )

                del rectified

            del candidate

        if candidates_tested >= MAX_CANDIDATES:
            break

    logging.info(
        "No barcode decoded candidates=%s opencv_detected=%s",
        candidates_tested,
        opencv_detected_count,
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
        "opencv_detected_count": opencv_detected_count,
        "formats_seen": formats_seen,
        "version": APP_VERSION,
        "note": (
            "No supported barcode payload was recovered from the supplied "
            "pixels. ZXing, Pyzbar and OpenCV QR detection were attempted."
        ),
    }
