import azure.functions as func
import logging
import os
import json
import base64
import uuid
import re
import numpy as np
import cv2
import fitz
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
AOAI_ENDPOINT = os.environ["AOAI_ENDPOINT"]
AOAI_KEY = os.environ["AOAI_KEY"]
AOAI_DEPLOYMENT = os.environ["AOAI_DEPLOYMENT"]
DI_ENDPOINT = os.environ["DI_ENDPOINT"]
DI_KEY = os.environ["DI_KEY"]
CONN_STR = os.environ["AzureWebJobsStorage"]

IN_CONTAINER = "idp-docs-in"
CLASS_CONTAINER = "idp-classification-out"
EXTRACT_CONTAINER = "idp-extraction-out"

# Minimum confidence required before an AI classification is accepted.
THRESHOLD = 90
OCR_TEXT_THRESHOLD = 90

# These are the only document type values that can be written to output JSON.
DOC_TYPE_INVOICE = "Invoices"
DOC_TYPE_COA = "Certificate of Acceptance"
DOC_TYPE_DDM = "Direct Debit Mandate"
DOC_TYPE_UBO = "UBO Declarations"
DOC_TYPE_ASSET = "Asset Registration Information"
DOC_TYPE_UNKNOWN = "Unknown"

ALLOWED_DOCUMENT_TYPES = {
    DOC_TYPE_INVOICE,
    DOC_TYPE_COA,
    DOC_TYPE_DDM,
    DOC_TYPE_UBO,
    DOC_TYPE_ASSET,
    DOC_TYPE_UNKNOWN,
}

# Approved QR document codes and their required output names.
QR_DOCUMENT_TYPE_MAP = {
    "COA": DOC_TYPE_COA,
    "DDM": DOC_TYPE_DDM,
    "UBO": DOC_TYPE_UBO,
}

# Normalizes small variations that the LLM may return.
DOCUMENT_TYPE_ALIASES = {
    "invoice": DOC_TYPE_INVOICE,
    "invoices": DOC_TYPE_INVOICE,
    "factuur": DOC_TYPE_INVOICE,
    "facturen": DOC_TYPE_INVOICE,
    "certificate of acceptance": DOC_TYPE_COA,
    "coa": DOC_TYPE_COA,
    "acceptatiecertificaat": DOC_TYPE_COA,
    "direct debit mandate": DOC_TYPE_DDM,
    "ddm": DOC_TYPE_DDM,
    "incassomachtiging": DOC_TYPE_DDM,
    "ubo declaration": DOC_TYPE_UBO,
    "ubo declarations": DOC_TYPE_UBO,
    "ubo": DOC_TYPE_UBO,
    "ubo verklaring": DOC_TYPE_UBO,
    "ubo verklaringen": DOC_TYPE_UBO,
    "asset registration information": DOC_TYPE_ASSET,
    "asset registration": DOC_TYPE_ASSET,
    "activaregistratie-informatie": DOC_TYPE_ASSET,
    "activaregistratie": DOC_TYPE_ASSET,
    "unknown": DOC_TYPE_UNKNOWN,
    "unknown document type": DOC_TYPE_UNKNOWN,
    "onbekend": DOC_TYPE_UNKNOWN,
}

client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_KEY,
    api_version="2024-07-01-preview",
)

di_client = DocumentIntelligenceClient(
    DI_ENDPOINT,
    AzureKeyCredential(DI_KEY),
)

blob_service = BlobServiceClient.from_connection_string(CONN_STR)

# Example: 1234567890-NL-COA-01
QR_PATTERN = re.compile(
    r"^(\d{10,11})-([A-Za-z]{2})-([A-Za-z0-9]+)-(\d+)$"
)
QR_TEXT_SEARCH_PATTERN = re.compile(
    r"\d{10,11}-[A-Za-z]{2}-[A-Za-z0-9]+-\d+"
)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def cors(resp: func.HttpResponse):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
    return resp


def strip_fences(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|```\s*$",
            "",
            text,
            flags=re.MULTILINE,
        ).strip()
    return text


def safe_number(value, default=0):
    try:
        number = float(value)
        return max(0, min(100, number))
    except (TypeError, ValueError):
        return default


def normalize_document_type(value):
    if not value:
        return DOC_TYPE_UNKNOWN

    cleaned = re.sub(r"\s+", " ", str(value)).strip()

    if cleaned in ALLOWED_DOCUMENT_TYPES:
        return cleaned

    return DOCUMENT_TYPE_ALIASES.get(cleaned.casefold(), DOC_TYPE_UNKNOWN)


def validate_llm_result(result):
    if not isinstance(result, dict):
        result = {}

    document_type = normalize_document_type(result.get("document_type"))
    classification_confidence = safe_number(
        result.get("classification_confidence"), 0
    )

    # Unknown classifications must not carry a positive accepted confidence.
    if document_type == DOC_TYPE_UNKNOWN:
        classification_confidence = 0

    fields = result.get("fields")
    if not isinstance(fields, dict):
        fields = {}

    field_confidence = result.get("field_confidence")
    if not isinstance(field_confidence, dict):
        field_confidence = {}

    clean_field_confidence = {
        str(key): safe_number(value, 0)
        for key, value in field_confidence.items()
    }

    return {
        "document_type": document_type,
        "classification_confidence": classification_confidence,
        "contract_number": result.get("contract_number"),
        "region": result.get("region"),
        "fields": fields,
        "field_confidence": clean_field_confidence,
        "extraction_confidence": safe_number(
            result.get("extraction_confidence"), 0
        ),
    }


# -----------------------------------------------------------------------------
# Image and QR helpers
# -----------------------------------------------------------------------------
def get_page_images(file_bytes, filename, dpi=200):
    if filename.lower().endswith(".pdf"):
        images = []
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                images.append(pix.tobytes("png"))
        finally:
            doc.close()
        return images

    # Normalize non-PDF input to PNG so the Azure OpenAI data URL MIME type
    # always matches the actual bytes.
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported or unreadable image file")

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Unable to convert the uploaded image to PNG")

    return [encoded.tobytes()]


def decode_qr_from_pages(page_images):
    detector = cv2.QRCodeDetector()

    for img_bytes in page_images:
        try:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for candidate in (gray, img):
                data, _points, _ = detector.detectAndDecode(candidate)
                if data:
                    return data.strip()

                retval, decoded_info, _points, _ = (
                    detector.detectAndDecodeMulti(candidate)
                )
                if retval:
                    for info in decoded_info:
                        if info:
                            return info.strip()
        except Exception as exc:
            logging.warning("QR decode failed on a page: %s", exc)

    return None


def find_qr_pattern_in_text(text):
    if not text:
        return None

    match = QR_TEXT_SEARCH_PATTERN.search(text)
    return match.group(0) if match else None


def parse_approved_qr(qr_value):
    """Parse QR data and return an approved classification when possible."""
    if not qr_value:
        return None

    match = QR_PATTERN.fullmatch(qr_value.strip())
    if not match:
        return None

    contract_number, region, doc_type_code, sequence_number = match.groups()
    mapped_type = QR_DOCUMENT_TYPE_MAP.get(doc_type_code.upper())

    if not mapped_type:
        return None

    return {
        "document_type": mapped_type,
        "classification_confidence": 99.5,
        "contract_number": contract_number,
        "region": region.upper(),
        "document_sequence": sequence_number,
        "qr_document_code": doc_type_code.upper(),
    }


# -----------------------------------------------------------------------------
# Azure AI processing
# -----------------------------------------------------------------------------
def analyze_with_document_intelligence(file_bytes):
    poller = di_client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=file_bytes),
    )
    result = poller.result()

    full_text = result.content or ""
    word_confidences = []

    for page in result.pages or []:
        for word in page.words or []:
            if word.confidence is not None:
                word_confidences.append(word.confidence)

    if word_confidences:
        ocr_confidence = (
            sum(word_confidences) / len(word_confidences)
        ) * 100
    else:
        ocr_confidence = 0

    return full_text, round(ocr_confidence, 2)


def classify_extract_with_llm(text=None, images_bytes=None):
    prompt = """
You are a strict document classification and extraction engine.
The document may be written in English, Dutch, or contain both languages.

Classify the document into exactly one of these allowed document_type values:
1. Invoices
2. Certificate of Acceptance
3. Direct Debit Mandate
4. UBO Declarations
5. Asset Registration Information
6. Unknown

Never create, invent, translate, shorten, pluralize, or return any other
document_type value.

Classification guidance:
- Invoices: invoice, supplier invoice, vendor invoice, billing document,
  factuur, leveranciersfactuur, verkoopfactuur, or a document requesting
  payment and normally containing invoice details such as invoice number,
  supplier, customer, line items, tax, amount, or payment terms.
- Certificate of Acceptance: COA, certificate confirming acceptance or
  delivery acceptance, acceptatiecertificaat, bewijs van acceptatie,
  acceptance confirmation, or a document where a customer accepts an asset,
  equipment, delivery, installation, or contractual item.
- Direct Debit Mandate: DDM, direct debit authorization, mandate for automatic
  collection, SEPA mandate, incassomachtiging, automatische incasso, or a
  document authorizing debit from a bank account.
- UBO Declarations: UBO declaration, ultimate beneficial owner declaration,
  beneficial ownership statement, UBO-verklaring, verklaring uiteindelijk
  belanghebbende, or a document identifying or declaring ultimate beneficial
  owners.
- Asset Registration Information: asset registration form, equipment or
  vehicle registration details, asset information sheet,
  activaregistratie, bedrijfsmiddelregistratie, objectregistratie, or a
  document used to register an asset and its identifying details.
- Unknown: use when the content does not clearly belong to one of the five
  business categories above. Do not force the nearest category.

Use the actual document meaning and layout, not language alone. A QR code may
be absent, unreadable, or not supplied to you, so classify from the available
content when necessary.

Also extract:
- contract_number, if present
- region as a 2-letter country code, if clearly present
- fields containing other important data relevant to the identified type
- field_confidence with a 0-100 confidence for each extracted field
- extraction_confidence from 0-100
- classification_confidence from 0-100

Return only one valid JSON object with exactly this top-level structure:
{
  "document_type": "Invoices | Certificate of Acceptance | Direct Debit Mandate | UBO Declarations | Asset Registration Information | Unknown",
  "classification_confidence": 0,
  "contract_number": null,
  "region": null,
  "fields": {},
  "field_confidence": {},
  "extraction_confidence": 0
}
""".strip()

    content = [{"type": "text", "text": prompt}]

    if text:
        content.append(
            {
                "type": "text",
                "text": f"Document text:\n{text[:12000]}",
            }
        )

    if images_bytes:
        for image_bytes in images_bytes:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}"
                    },
                }
            )

    response = client.chat.completions.create(
        model=AOAI_DEPLOYMENT,
        messages=[{"role": "user", "content": content}],
        max_tokens=1500,
        temperature=0,
    )

    raw = strip_fences(response.choices[0].message.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("LLM output was not valid JSON")
        parsed = {}

    return validate_llm_result(parsed)


# -----------------------------------------------------------------------------
# Main document pipeline
# -----------------------------------------------------------------------------
def process_document(filename, file_bytes):
    page_images = get_page_images(file_bytes, filename, dpi=200)
    qr_scan_images = get_page_images(file_bytes, filename, dpi=350)

    # First attempt: decode the QR image directly.
    qr_value = decode_qr_from_pages(qr_scan_images)
    qr_source = "barcode_image" if qr_value else None

    # OCR is always performed because it is required for extraction and for
    # content classification when an approved QR is not available.
    full_text, ocr_confidence = analyze_with_document_intelligence(file_bytes)

    # Second attempt: locate the expected QR-style reference in OCR text.
    if not qr_value:
        text_match = find_qr_pattern_in_text(full_text)
        if text_match:
            qr_value = text_match
            qr_source = "text_fallback"

    approved_qr = parse_approved_qr(qr_value)

    # Classify and extract from OCR text when OCR quality is high. Otherwise,
    # send document page images to Azure OpenAI Vision.
    if ocr_confidence >= OCR_TEXT_THRESHOLD:
        llm_result = classify_extract_with_llm(text=full_text)
        extraction_source = "document_intelligence_text"
    else:
        llm_result = classify_extract_with_llm(images_bytes=page_images)
        extraction_source = "azure_openai_vision_fallback"

    extraction = {
        "fields": llm_result.get("fields", {}),
        "field_confidence": llm_result.get("field_confidence", {}),
        "overall_extraction_confidence": llm_result.get(
            "extraction_confidence", 0
        ),
        "extraction_source": extraction_source,
        "document_intelligence_ocr_confidence": ocr_confidence,
    }

    if approved_qr:
        # An approved COA, DDM, or UBO QR code is authoritative and overrides
        # the AI document type. AI is still used for field extraction.
        classification = {
            "qr_identification": "Yes",
            "qr_raw_value": qr_value,
            "qr_source": qr_source,
            "qr_document_code": approved_qr["qr_document_code"],
            "document_sequence": approved_qr["document_sequence"],
            "classification_source": "approved_qr_mapping",
            "document_type": approved_qr["document_type"],
            "classification_confidence": approved_qr[
                "classification_confidence"
            ],
            "meets_threshold": True,
            "contract_number": approved_qr["contract_number"],
            "region": approved_qr["region"],
        }
    else:
        # If there is no approved QR mapping, classify from document content.
        # This also covers a missing/unreadable QR and an unsupported QR code.
        ai_type = normalize_document_type(
            llm_result.get("document_type")
        )
        ai_confidence = safe_number(
            llm_result.get("classification_confidence"), 0
        )
        meets_threshold = (
            ai_type != DOC_TYPE_UNKNOWN
            and ai_confidence >= THRESHOLD
        )
        final_type = ai_type if meets_threshold else DOC_TYPE_UNKNOWN

        qr_pattern_valid = bool(
            qr_value and QR_PATTERN.fullmatch(qr_value.strip())
        )

        classification = {
            "qr_identification": "Yes" if qr_value else "No",
            "qr_raw_value": qr_value,
            "qr_source": qr_source,
            "qr_approved_type": False if qr_value else None,
            "qr_pattern_valid": qr_pattern_valid if qr_value else None,
            "classification_source": extraction_source,
            "document_type": final_type,
            "classification_confidence": (
                ai_confidence if meets_threshold else 0
            ),
            "meets_threshold": meets_threshold,
            "contract_number": llm_result.get("contract_number"),
            "region": llm_result.get("region"),
        }

    return classification, extraction


# -----------------------------------------------------------------------------
# Blob-triggered document processing
# -----------------------------------------------------------------------------
@app.blob_trigger(
    arg_name="myblob",
    path="idp-docs-in/{name}",
    connection="AzureWebJobsStorage",
    source="EventGrid",
)
def DocumentClassifier(myblob: func.InputStream):
    filename = myblob.name.split("/")[-1]
    doc_id = filename.rsplit(".", 1)[0]

    logging.info("Processing document: %s", filename)
    file_bytes = myblob.read()

    try:
        classification, extraction = process_document(
            filename,
            file_bytes,
        )
    except Exception as exc:
        logging.exception("Processing failed for %s", filename)
        classification = {
            "qr_identification": "No",
            "qr_raw_value": None,
            "qr_source": None,
            "classification_source": "processing_error",
            "document_type": DOC_TYPE_UNKNOWN,
            "classification_confidence": 0,
            "meets_threshold": False,
            "contract_number": None,
            "region": None,
            "error": str(exc),
        }
        extraction = {
            "fields": {},
            "field_confidence": {},
            "overall_extraction_confidence": 0,
            "extraction_source": "processing_error",
        }

    blob_service.get_blob_client(
        CLASS_CONTAINER,
        f"{doc_id}.json",
    ).upload_blob(
        json.dumps(classification, indent=2, ensure_ascii=False),
        overwrite=True,
    )

    blob_service.get_blob_client(
        EXTRACT_CONTAINER,
        f"{doc_id}.json",
    ).upload_blob(
        json.dumps(extraction, indent=2, ensure_ascii=False),
        overwrite=True,
    )

    logging.info(
        "Saved classification and extraction for %s",
        doc_id,
    )


# -----------------------------------------------------------------------------
# HTTP APIs
# -----------------------------------------------------------------------------
@app.route(route="documents", methods=["GET"])
def list_documents(req: func.HttpRequest) -> func.HttpResponse:
    results = []
    container = blob_service.get_container_client(CLASS_CONTAINER)

    for blob in container.list_blobs():
        doc_id = blob.name.rsplit(".", 1)[0]
        try:
            data = (
                blob_service.get_blob_client(
                    CLASS_CONTAINER,
                    blob.name,
                )
                .download_blob()
                .readall()
            )
            classification = json.loads(data)
        except Exception as exc:
            logging.warning(
                "Unable to read classification blob %s: %s",
                blob.name,
                exc,
            )
            classification = {}

        results.append(
            {
                "id": doc_id,
                "name": blob.name,
                **classification,
            }
        )

    return cors(
        func.HttpResponse(
            json.dumps(results, ensure_ascii=False),
            mimetype="application/json",
        )
    )


@app.route(route="documents/{id}", methods=["GET"])
def get_document(req: func.HttpRequest) -> func.HttpResponse:
    doc_id = req.route_params.get("id")

    try:
        classification_data = (
            blob_service.get_blob_client(
                CLASS_CONTAINER,
                f"{doc_id}.json",
            )
            .download_blob()
            .readall()
        )
        classification = json.loads(classification_data)
    except Exception:
        return cors(
            func.HttpResponse(
                json.dumps({"error": "not found"}),
                status_code=404,
                mimetype="application/json",
            )
        )

    try:
        extraction_data = (
            blob_service.get_blob_client(
                EXTRACT_CONTAINER,
                f"{doc_id}.json",
            )
            .download_blob()
            .readall()
        )
        extraction = json.loads(extraction_data)
    except Exception:
        extraction = {
            "fields": {},
            "field_confidence": {},
            "overall_extraction_confidence": 0,
        }

    response = {
        "classification": classification,
        "extraction": extraction,
    }

    return cors(
        func.HttpResponse(
            json.dumps(response, ensure_ascii=False),
            mimetype="application/json",
        )
    )


@app.route(route="documents/{id}/file", methods=["GET"])
def get_document_file(req: func.HttpRequest) -> func.HttpResponse:
    doc_id = req.route_params.get("id")
    container_client = blob_service.get_container_client(IN_CONTAINER)

    supported_files = [
        (".pdf", "application/pdf"),
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
    ]

    for extension, content_type in supported_files:
        blob_client = container_client.get_blob_client(
            f"{doc_id}{extension}"
        )
        if blob_client.exists():
            return cors(
                func.HttpResponse(
                    blob_client.download_blob().readall(),
                    mimetype=content_type,
                )
            )

    return cors(
        func.HttpResponse(
            json.dumps({"error": "file not found"}),
            status_code=404,
            mimetype="application/json",
        )
    )


@app.route(route="documents/upload", methods=["POST"])
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    file_bytes = req.get_body()

    if not file_bytes:
        return cors(
            func.HttpResponse(
                json.dumps({"error": "empty file"}),
                status_code=400,
                mimetype="application/json",
            )
        )

    requested_filename = req.params.get("filename")
    filename = os.path.basename(
        requested_filename or f"{uuid.uuid4()}.pdf"
    )

    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    extension = os.path.splitext(filename)[1].lower()

    if extension not in allowed_extensions:
        return cors(
            func.HttpResponse(
                json.dumps(
                    {
                        "error": (
                            "unsupported file type; allowed types are "
                            "PDF, PNG, JPG, and JPEG"
                        )
                    }
                ),
                status_code=400,
                mimetype="application/json",
            )
        )

    blob_service.get_blob_client(
        IN_CONTAINER,
        filename,
    ).upload_blob(
        file_bytes,
        overwrite=True,
    )

    doc_id = filename.rsplit(".", 1)[0]
    response = {
        "status": "uploaded",
        "processing_status": "pending",
        "document_id": doc_id,
        "filename": filename,
    }

    return cors(
        func.HttpResponse(
            json.dumps(response),
            mimetype="application/json",
        )
    )
