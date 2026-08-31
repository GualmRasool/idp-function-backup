import azure.functions as func
import base64
import json
import logging
import os
import re
import uuid

import cv2
import fitz
import numpy as np
import requests
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
APP_VERSION = "1.4.0"

# Configuration
AOAI_ENDPOINT = os.environ["AOAI_ENDPOINT"]
AOAI_KEY = os.environ["AOAI_KEY"]
AOAI_DEPLOYMENT = os.environ["AOAI_DEPLOYMENT"]
DI_ENDPOINT = os.environ["DI_ENDPOINT"]
DI_KEY = os.environ["DI_KEY"]
CONN_STR = os.environ["AzureWebJobsStorage"]
QR_DECODER_URL = os.environ.get("QR_DECODER_URL", "").strip().strip('"').strip("'").rstrip("/")
QR_DECODER_KEY = os.environ.get("QR_DECODER_KEY", "").strip().strip('"').strip("'")
QR_DECODER_CONNECT_TIMEOUT_SECONDS = 10
QR_DECODER_READ_TIMEOUT_SECONDS = 90
BARCODE_SCAN_DPI = int(os.environ.get("BARCODE_SCAN_DPI", "350"))
CONTENT_BOUNDARY_CONFIDENCE = float(os.environ.get("CONTENT_BOUNDARY_CONFIDENCE", "80"))
CONTENT_BOUNDARY_MAX_TEXT = int(os.environ.get("CONTENT_BOUNDARY_MAX_TEXT", "5000"))

IN_CONTAINER = "idp-docs-in"
SPLIT_CONTAINER = "idp-docs-split"
CLASS_CONTAINER = "idp-classification-out"
EXTRACT_CONTAINER = "idp-extraction-out"
MANIFEST_CONTAINER = "idp-split-manifest"
THRESHOLD = 90
OCR_TEXT_THRESHOLD = 90

DOC_TYPE_INVOICE = "Invoices"
DOC_TYPE_COA = "Certificate of Acceptance"
DOC_TYPE_DDM = "Direct Debit Mandate"
DOC_TYPE_UBO = "UBO Declarations"
DOC_TYPE_ASSET = "Asset Registration Information"
DOC_TYPE_UNKNOWN = "Unknown"
ALLOWED_DOCUMENT_TYPES = {DOC_TYPE_INVOICE, DOC_TYPE_COA, DOC_TYPE_DDM, DOC_TYPE_UBO, DOC_TYPE_ASSET, DOC_TYPE_UNKNOWN}
QR_DOCUMENT_TYPE_MAP = {"COA": DOC_TYPE_COA, "DDM": DOC_TYPE_DDM, "UBO": DOC_TYPE_UBO}
DOCUMENT_TYPE_ALIASES = {
    "invoice": DOC_TYPE_INVOICE, "invoices": DOC_TYPE_INVOICE, "factuur": DOC_TYPE_INVOICE,
    "certificate of acceptance": DOC_TYPE_COA, "coa": DOC_TYPE_COA,
    "direct debit mandate": DOC_TYPE_DDM, "ddm": DOC_TYPE_DDM,
    "ubo declaration": DOC_TYPE_UBO, "ubo declarations": DOC_TYPE_UBO, "ubo": DOC_TYPE_UBO,
    "asset registration information": DOC_TYPE_ASSET, "asset registration": DOC_TYPE_ASSET,
    "unknown": DOC_TYPE_UNKNOWN,
}
QR_PATTERN = re.compile(r"^(\d{10,11})-([A-Za-z]{2})-([A-Za-z0-9]+)-(\d+)$")
QR_TEXT_SEARCH_PATTERN = re.compile(r"\d{10,11}-[A-Za-z]{2}-[A-Za-z0-9]+-\d+")

client = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version="2024-07-01-preview")
di_client = DocumentIntelligenceClient(DI_ENDPOINT, AzureKeyCredential(DI_KEY))
blob_service = BlobServiceClient.from_connection_string(CONN_STR)


def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
    return resp


def ensure_containers():
    for name in (SPLIT_CONTAINER, CLASS_CONTAINER, EXTRACT_CONTAINER, MANIFEST_CONTAINER):
        try:
            blob_service.create_container(name)
        except ResourceExistsError:
            pass


def strip_fences(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|```\s*$", "", text, flags=re.MULTILINE).strip()
    return text


def safe_number(value, default=0):
    try:
        return max(0, min(100, float(value)))
    except (TypeError, ValueError):
        return default

def normalize_confidence(value, default=0):
    """
    Normalize confidence values to a 0-100 scale.

    Supports models returning:
    - 0.95
    - 95
    - "0.95"
    - "95"
    """
    try:
        confidence = float(value)

        if 0 <= confidence <= 1:
            confidence *= 100

        return round(
            max(0, min(100, confidence)),
            2,
        )

    except (TypeError, ValueError):
        return default

def normalize_document_type(value):
    if not value:
        return DOC_TYPE_UNKNOWN
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    if cleaned in ALLOWED_DOCUMENT_TYPES:
        return cleaned
    return DOCUMENT_TYPE_ALIASES.get(cleaned.casefold(), DOC_TYPE_UNKNOWN)


def clean_barcode_value(value):
    return str(value).replace("\x00", "").replace("<NUL>", "").strip()


def validate_llm_result(result):
    """
    Validate and normalize the final classification and extraction result.

    Confidence values may be returned as either:
    - 0.95
    - 95
    - "0.95"
    - "95"

    All confidence values are normalized to a 0-100 scale.
    """
    if not isinstance(result, dict):
        result = {}

    document_type = normalize_document_type(
        result.get("document_type")
    )

    classification_confidence = normalize_confidence(
        result.get("classification_confidence"),
        0,
    )

    if document_type == DOC_TYPE_UNKNOWN:
        classification_confidence = 0

    fields = result.get("fields")

    if not isinstance(fields, dict):
        fields = {}

    field_confidence = result.get("field_confidence")

    if not isinstance(field_confidence, dict):
        field_confidence = {}

    normalized_field_confidence = {
        str(key): normalize_confidence(value, 0)
        for key, value in field_confidence.items()
    }

    return {
        "document_type": document_type,
        "classification_confidence":
            classification_confidence,
        "contract_number": result.get(
            "contract_number"
        ),
        "region": result.get("region"),
        "fields": fields,
        "field_confidence":
            normalized_field_confidence,
        "extraction_confidence":
            normalize_confidence(
                result.get(
                    "extraction_confidence"
                ),
                0,
            ),
    }


def analyze_with_document_intelligence(file_bytes):
    poller = di_client.begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(bytes_source=file_bytes))
    result = poller.result()
    text = result.content or ""
    scores = [w.confidence for p in (result.pages or []) for w in (p.words or []) if w.confidence is not None]
    confidence = (sum(scores) / len(scores) * 100) if scores else 0
    return text, round(confidence, 2)


def render_page(page, dpi=BARCODE_SCAN_DPI):
    return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")


def decode_barcode_service(image_bytes, image_name):
    if not QR_DECODER_URL or not QR_DECODER_KEY:
        return {"found": False, "values": [], "decoder": None, "barcode_type": None, "candidate": None}
    try:
        response = requests.post(
            f"{QR_DECODER_URL}/decode?mode=fast",
            headers={"x-api-key": QR_DECODER_KEY},
            files={"file": (image_name, image_bytes, "image/png")},
            timeout=(QR_DECODER_CONNECT_TIMEOUT_SECONDS, QR_DECODER_READ_TIMEOUT_SECONDS),
        )
        logging.info("Barcode response image=%s status=%s body=%s", image_name, response.status_code, response.text[:800])
        response.raise_for_status()
        result = response.json()
        values = result.get("qr_values", [])
        values = [clean_barcode_value(v) for v in values if v]
        return {
            "found": bool(result.get("found") and values), "values": values,
            "decoder": result.get("decoder"), "barcode_type": result.get("barcode_type"),
            "candidate": result.get("candidate"),
        }
    except Exception as exc:
        logging.warning("Barcode request failed image=%s error=%s", image_name, exc)
        return {"found": False, "values": [], "decoder": None, "barcode_type": None, "candidate": None}


def scan_pdf_pages_for_barcodes(file_bytes):
    results = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for index in range(doc.page_count):
            page_no = index + 1
            decoded = decode_barcode_service(render_page(doc[index]), f"page_{page_no:04d}.png")
            results.append({
                "page_index": index, "page_number": page_no, "found": decoded["found"],
                "barcode_values": decoded["values"],
                "barcode_value": decoded["values"][0] if decoded["values"] else None,
                "barcode_type": decoded.get("barcode_type"), "decoder": decoded.get("decoder"),
                "candidate": decoded.get("candidate"),
            })
    finally:
        doc.close()
    return results


def single_page_pdf_bytes(source_doc, page_index):
    out = fitz.open()
    try:
        out.insert_pdf(source_doc, from_page=page_index, to_page=page_index)
        return out.tobytes(garbage=4, deflate=True)
    finally:
        out.close()


def classify_page_for_boundary(page_text):
    prompt = """Classify one page in a mixed document batch into exactly one type: Invoices, Certificate of Acceptance, Direct Debit Mandate, UBO Declarations, Asset Registration Information, Unknown. Identify whether it is first_page, continuation, final_page, single_page, or unknown. Extract contract_number, invoice_number, document_number, and reference_number when visible. Return JSON only with document_type, confidence, page_role, document_keys."""
    response = client.chat.completions.create(
        model=AOAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt + "\n\nPage text:\n" + page_text[:CONTENT_BOUNDARY_MAX_TEXT]}],
        max_tokens=500, temperature=0, response_format={"type": "json_object"},
    )
    try:
        result = json.loads(strip_fences(response.choices[0].message.content))
    except json.JSONDecodeError:
        result = {}
    keys = result.get("document_keys") if isinstance(result.get("document_keys"), dict) else {}
    return {
        "document_type": normalize_document_type(result.get("document_type")),
        "confidence": normalize_confidence(result.get("confidence"), 0),
        "page_role": str(result.get("page_role") or "unknown"),
        "document_keys": keys,
    }


def analyze_pages_for_mixed_splitting(file_bytes, barcode_results):
    profiles = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for result in barcode_results:
            page_pdf = single_page_pdf_bytes(doc, result["page_index"])
            text, ocr_conf = analyze_with_document_intelligence(page_pdf)
            profile = classify_page_for_boundary(text)
            profile.update({
                "page_index": result["page_index"], "page_number": result["page_number"],
                "text": text, "ocr_confidence": ocr_conf, "barcode_found": result["found"],
                "barcode_value": result.get("barcode_value"), "barcode_type": result.get("barcode_type"),
            })
            profiles.append(profile)
    finally:
        doc.close()
    return profiles


def decide_content_boundary(previous, current):
    if (
        previous["document_type"] != DOC_TYPE_UNKNOWN and current["document_type"] != DOC_TYPE_UNKNOWN
        and previous["document_type"] != current["document_type"]
        and previous["confidence"] >= CONTENT_BOUNDARY_CONFIDENCE
        and current["confidence"] >= CONTENT_BOUNDARY_CONFIDENCE
    ):
        return {"starts_new_document": True, "confidence": min(previous["confidence"], current["confidence"]), "reason": "high_confidence_document_type_change"}
    if current["page_role"] in {"first_page", "single_page"} and current["confidence"] >= CONTENT_BOUNDARY_CONFIDENCE:
        return {"starts_new_document": True, "confidence": current["confidence"], "reason": "current_page_identified_as_first_page"}

    prompt = """Determine whether CURRENT starts a new logical document or continues PREVIOUS. Consider type, title, new identifiers, page numbering and continuation wording. Two adjacent documents may have the same type. Return JSON only: {\"starts_new_document\":false,\"confidence\":0,\"reason\":\"...\"}."""
    payload = {
        "previous": {"type": previous["document_type"], "role": previous["page_role"], "keys": previous["document_keys"], "text": previous["text"][:CONTENT_BOUNDARY_MAX_TEXT]},
        "current": {"type": current["document_type"], "role": current["page_role"], "keys": current["document_keys"], "text": current["text"][:CONTENT_BOUNDARY_MAX_TEXT]},
    }
    response = client.chat.completions.create(
        model=AOAI_DEPLOYMENT, messages=[{"role": "user", "content": prompt + "\n\n" + json.dumps(payload)}],
        max_tokens=300, temperature=0, response_format={"type": "json_object"},
    )
    try:
        result = json.loads(strip_fences(response.choices[0].message.content))
    except json.JSONDecodeError:
        result = {}
    confidence = normalize_confidence(result.get("confidence"), 0)
    return {
        "starts_new_document": bool(result.get("starts_new_document")) and confidence >= CONTENT_BOUNDARY_CONFIDENCE,
        "confidence": confidence,
        "reason": str(result.get("reason") or "low_confidence_continuation"),
    }


def build_hybrid_ranges(barcodes, profiles, total_pages):
    ranges, start = [], 0
    pending = None
    for current in range(total_pages):
        if current > start:
            boundary = decide_content_boundary(profiles[current - 1], profiles[current])
            if boundary["starts_new_document"]:
                ranges.append({
                    "start_index": start, "end_index": current - 1,
                    "start_page": start + 1, "end_page": current,
                    "boundary_barcode": None, "boundary_barcode_type": None,
                    "boundary_decoder": None, "boundary_candidate": None,
                    "boundary_page": None, "has_barcode_boundary": False,
                    "split_method": "content_boundary",
                    "content_boundary_confidence": boundary["confidence"],
                    "content_boundary_reason": boundary["reason"],
                })
                start, pending = current, boundary
        if barcodes[current]["found"]:
            ranges.append({
                "start_index": start, "end_index": current,
                "start_page": start + 1, "end_page": current + 1,
                "boundary_barcode": barcodes[current].get("barcode_value"),
                "boundary_barcode_type": barcodes[current].get("barcode_type"),
                "boundary_decoder": barcodes[current].get("decoder"),
                "boundary_candidate": barcodes[current].get("candidate"),
                "boundary_page": current + 1, "has_barcode_boundary": True,
                "split_method": "barcode_end_marker",
                "content_boundary_confidence": pending["confidence"] if pending else None,
                "content_boundary_reason": pending["reason"] if pending else None,
            })
            start, pending = current + 1, None
    if start < total_pages:
        ranges.append({
            "start_index": start, "end_index": total_pages - 1,
            "start_page": start + 1, "end_page": total_pages,
            "boundary_barcode": None, "boundary_barcode_type": None,
            "boundary_decoder": None, "boundary_candidate": None,
            "boundary_page": None, "has_barcode_boundary": False,
            "split_method": "content_or_trailing_segment",
            "content_boundary_confidence": pending["confidence"] if pending else None,
            "content_boundary_reason": pending["reason"] if pending else "end_of_batch",
        })
    return ranges


def extract_pdf_range(file_bytes, start_index, end_index):
    source = fitz.open(stream=file_bytes, filetype="pdf")
    output = fitz.open()
    try:
        output.insert_pdf(source, from_page=start_index, to_page=end_index)
        return output.tobytes(garbage=4, deflate=True)
    finally:
        output.close(); source.close()


def split_pdf_hybrid(filename, file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        total = doc.page_count
    finally:
        doc.close()
    barcodes = scan_pdf_pages_for_barcodes(file_bytes)
    profiles = analyze_pages_for_mixed_splitting(file_bytes, barcodes)
    ranges = build_hybrid_ranges(barcodes, profiles, total)
    stem = os.path.splitext(os.path.basename(filename))[0]
    segments = []
    for number, page_range in enumerate(ranges, 1):
        segment_id = f"{stem}__doc_{number:03d}"
        split_name = f"{segment_id}__pages_{page_range['start_page']:03d}-{page_range['end_page']:03d}.pdf"
        segments.append({
            "document_number": number, "segment_id": segment_id, "filename": split_name,
            "file_bytes": extract_pdf_range(file_bytes, page_range["start_index"], page_range["end_index"]),
            **page_range,
        })
    return segments, barcodes, profiles


def get_page_images(file_bytes, filename, dpi=200):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            return [p.get_pixmap(dpi=dpi, alpha=False).tobytes("png") for p in doc]
        finally:
            doc.close()
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable image")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Unable to encode image")
    return [encoded.tobytes()]


def classify_extract_with_llm(text=None, images_bytes=None):
    prompt = """
You are a strict document classification and extraction engine.

Classify the complete logical document into exactly one of these values:

1. Invoices
2. Certificate of Acceptance
3. Direct Debit Mandate
4. UBO Declarations
5. Asset Registration Information
6. Unknown

Classification guidance:

- Invoices:
  Invoice, supplier invoice, vendor invoice, factuur,
  billing document, line items, totals, VAT, due date,
  invoice number, or payment terms.

- Certificate of Acceptance:
  COA, certificate confirming delivery, installation,
  acceptance testing, or customer acceptance.

- Direct Debit Mandate:
  DDM, SEPA mandate, bank debit authorization,
  incassomachtiging, or automatic debit authorization.

- UBO Declarations:
  Ultimate beneficial owner declaration, UBO register,
  beneficial ownership statement, or UBO-verklaring.

- Asset Registration Information:
  Asset registration form, fixed asset registration,
  machinery registration, equipment registration,
  serial number, depreciation, asset ledger, or
  maintenance schedule.

- Unknown:
  Use only when the document clearly does not belong
  to one of the five supported business types.

Important confidence rule:

Return every confidence value on a 0-100 scale.

Correct:
  95

Incorrect:
  0.95

Extract:

- contract_number, invoice_number, registration number,
  or primary document reference when present
- region as a 2-letter country code when clearly known
- other important document fields
- field_confidence for every extracted field
- extraction_confidence
- classification_confidence

Return only one JSON object:

{
  "document_type": "Invoices",
  "classification_confidence": 95,
  "contract_number": null,
  "region": "NL",
  "fields": {},
  "field_confidence": {},
  "extraction_confidence": 95
}
""".strip()
    content = [{"type": "text", "text": prompt}]
    if text:
        content.append({"type": "text", "text": "Document text:\n" + text[:12000]})
    if images_bytes:
        for image in images_bytes:
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("utf-8")}})
    response = client.chat.completions.create(model=AOAI_DEPLOYMENT, messages=[{"role": "user", "content": content}], max_tokens=1500, temperature=0, response_format={"type": "json_object"})
    try:
        parsed = json.loads(strip_fences(response.choices[0].message.content))
    except json.JSONDecodeError:
        parsed = {}
    return validate_llm_result(parsed)


def parse_approved_qr(value):
    if not value:
        return None
    match = QR_PATTERN.fullmatch(clean_barcode_value(value))
    if not match:
        return None
    contract, region, code, sequence = match.groups()
    mapped = QR_DOCUMENT_TYPE_MAP.get(code.upper())
    if not mapped:
        return None
    return {"document_type": mapped, "classification_confidence": 99.5, "contract_number": contract, "region": region.upper(), "document_sequence": sequence, "qr_document_code": code.upper()}


def process_logical_document(filename, file_bytes, barcode=None, barcode_type=None, barcode_decoder=None):
    text, ocr_conf = analyze_with_document_intelligence(file_bytes)
    approved = parse_approved_qr(barcode)
    if ocr_conf >= OCR_TEXT_THRESHOLD:
        llm = classify_extract_with_llm(text=text)
        source = "document_intelligence_text"
    else:
        llm = classify_extract_with_llm(images_bytes=get_page_images(file_bytes, filename))
        source = "azure_openai_vision_fallback"
    extraction = {"fields": llm["fields"], "field_confidence": llm["field_confidence"], "overall_extraction_confidence": llm["extraction_confidence"], "extraction_source": source, "document_intelligence_ocr_confidence": ocr_conf}
    if approved:
        classification = {
            "qr_identification": "Yes", "qr_raw_value": barcode, "qr_source": "boundary_page_barcode",
            "barcode_type": barcode_type, "barcode_decoder": barcode_decoder,
            "qr_document_code": approved["qr_document_code"], "document_sequence": approved["document_sequence"],
            "classification_source": "approved_qr_mapping", "document_type": approved["document_type"],
            "classification_confidence": approved["classification_confidence"], "meets_threshold": True,
            "contract_number": approved["contract_number"], "region": approved["region"],
        }
    else:
        doc_type, confidence = llm["document_type"], llm["classification_confidence"]
        meets = doc_type != DOC_TYPE_UNKNOWN and confidence >= THRESHOLD
        classification = {
            "qr_identification": "Yes" if barcode else "No", "qr_raw_value": barcode,
            "qr_source": "boundary_page_barcode" if barcode else None,
            "barcode_type": barcode_type, "barcode_decoder": barcode_decoder,
            "qr_approved_type": False if barcode else None,
            "classification_source": source, "document_type": doc_type if meets else DOC_TYPE_UNKNOWN,
            "classification_confidence": confidence if meets else 0, "meets_threshold": meets,
            "contract_number": llm["contract_number"], "region": llm["region"],
        }
    return classification, extraction


def process_combined_pdf(filename, file_bytes):
    ensure_containers()
    segments, barcode_results, profiles = split_pdf_hybrid(filename, file_bytes)
    documents = []
    for segment in segments:
        sid, split_name, split_bytes = segment["segment_id"], segment["filename"], segment["file_bytes"]
        blob_service.get_blob_client(SPLIT_CONTAINER, split_name).upload_blob(split_bytes, overwrite=True, content_type="application/pdf")
        classification, extraction = process_logical_document(
            split_name, split_bytes, segment["boundary_barcode"],
            segment["boundary_barcode_type"], segment["boundary_decoder"],
        )
        metadata = {
            "source_file": filename, "split_document_id": sid, "split_document_file": split_name,
            "document_number": segment["document_number"], "start_page": segment["start_page"],
            "end_page": segment["end_page"], "page_count": segment["end_page"] - segment["start_page"] + 1,
            "boundary_page": segment["boundary_page"], "has_barcode_boundary": segment["has_barcode_boundary"],
            "split_method": segment["split_method"],
            "content_boundary_confidence": segment["content_boundary_confidence"],
            "content_boundary_reason": segment["content_boundary_reason"],
        }
        classification.update(metadata)
        extraction.update({"source_file": filename, "split_document_id": sid, "start_page": segment["start_page"], "end_page": segment["end_page"]})
        blob_service.get_blob_client(CLASS_CONTAINER, sid + ".json").upload_blob(json.dumps(classification, indent=2), overwrite=True, content_type="application/json")
        blob_service.get_blob_client(EXTRACT_CONTAINER, sid + ".json").upload_blob(json.dumps(extraction, indent=2), overwrite=True, content_type="application/json")
        documents.append({**metadata, "document_id": sid, "document_type": classification["document_type"], "classification_source": classification["classification_source"], "boundary_barcode": segment["boundary_barcode"]})

    manifest = {
        "source_file": filename, "total_pages": len(barcode_results),
        "split_rule": "hybrid_barcode_end_marker_and_content_boundary",
        "barcode_pages": [p["page_number"] for p in barcode_results if p["found"]],
        "document_count": len(documents), "documents": documents,
        "page_barcode_results": barcode_results,
        "page_content_profiles": [{k: v for k, v in p.items() if k != "text"} for p in profiles],
    }
    stem = os.path.splitext(os.path.basename(filename))[0]
    blob_service.get_blob_client(MANIFEST_CONTAINER, stem + ".json").upload_blob(json.dumps(manifest, indent=2), overwrite=True, content_type="application/json")
    return manifest


@app.blob_trigger(arg_name="myblob", path="idp-docs-in/{name}", connection="AzureWebJobsStorage", source="EventGrid")
def DocumentClassifier(myblob: func.InputStream):
    filename = myblob.name.split("/")[-1]
    file_bytes = myblob.read()
    if filename.lower().endswith(".pdf"):
        manifest = process_combined_pdf(filename, file_bytes)
        logging.info("Processed mixed batch file=%s documents=%s", filename, manifest["document_count"])
    else:
        raise ValueError("Mixed batch splitter currently expects PDF input")


@app.route(route="documents", methods=["GET"])
def list_documents(req: func.HttpRequest) -> func.HttpResponse:
    results = []
    for blob in blob_service.get_container_client(CLASS_CONTAINER).list_blobs():
        try:
            data = blob_service.get_blob_client(CLASS_CONTAINER, blob.name).download_blob().readall()
            results.append({"id": blob.name.rsplit(".", 1)[0], **json.loads(data)})
        except Exception as exc:
            logging.warning("Unable to read %s: %s", blob.name, exc)
    return cors(func.HttpResponse(json.dumps(results), mimetype="application/json"))


@app.route(route="documents/{id}", methods=["GET"])
def get_document(req: func.HttpRequest) -> func.HttpResponse:
    doc_id = req.route_params.get("id")
    try:
        classification = json.loads(blob_service.get_blob_client(CLASS_CONTAINER, doc_id + ".json").download_blob().readall())
        extraction = json.loads(blob_service.get_blob_client(EXTRACT_CONTAINER, doc_id + ".json").download_blob().readall())
        return cors(func.HttpResponse(json.dumps({"classification": classification, "extraction": extraction}), mimetype="application/json"))
    except Exception:
        return cors(func.HttpResponse(json.dumps({"error": "not found"}), status_code=404, mimetype="application/json"))


@app.route(route="documents/upload", methods=["POST"])
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    data = req.get_body()
    if not data:
        return cors(func.HttpResponse(json.dumps({"error": "empty file"}), status_code=400, mimetype="application/json"))
    filename = os.path.basename(req.params.get("filename") or f"{uuid.uuid4()}.pdf")
    if not filename.lower().endswith(".pdf"):
        return cors(func.HttpResponse(json.dumps({"error": "mixed batch upload requires PDF"}), status_code=400, mimetype="application/json"))
    blob_service.get_blob_client(IN_CONTAINER, filename).upload_blob(data, overwrite=True)
    return cors(func.HttpResponse(json.dumps({"status": "uploaded", "processing_status": "pending", "filename": filename, "version": APP_VERSION}), mimetype="application/json"))
