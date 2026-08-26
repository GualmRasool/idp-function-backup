import azure.functions as func
import logging, os, json, base64, uuid, re, time
import numpy as np
import cv2
import fitz
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

AOAI_ENDPOINT = os.environ["AOAI_ENDPOINT"]
AOAI_KEY = os.environ["AOAI_KEY"]
AOAI_DEPLOYMENT = os.environ["AOAI_DEPLOYMENT"]
DI_ENDPOINT = os.environ["DI_ENDPOINT"]
DI_KEY = os.environ["DI_KEY"]
CONN_STR = os.environ["AzureWebJobsStorage"]

IN_CONTAINER = "idp-docs-in"
CLASS_CONTAINER = "idp-classification-out"
EXTRACT_CONTAINER = "idp-extraction-out"
THRESHOLD = 90
OCR_TEXT_THRESHOLD = 90

DOC_TYPE_INVOICE = "Invoices"
DOC_TYPE_COA = "Certificate of Acceptance"
DOC_TYPE_DDM = "Direct Debit Mandate"
DOC_TYPE_UBO = "UBO Declarations"
DOC_TYPE_CONTRACTS = "Contracts"
DOC_TYPE_ASSET = "Asset Registration Information"
DOC_TYPE_UNKNOWN = "Unknown"

ALLOWED_DOCUMENT_TYPES = {
    DOC_TYPE_INVOICE, DOC_TYPE_COA, DOC_TYPE_DDM, DOC_TYPE_UBO,
    DOC_TYPE_CONTRACTS, DOC_TYPE_ASSET, DOC_TYPE_UNKNOWN
}

QR_DOCUMENT_TYPE_MAP = {
    "COA": DOC_TYPE_COA,
    "DDM": DOC_TYPE_DDM,
    "UBO": DOC_TYPE_UBO,
    "CONTRACT": DOC_TYPE_CONTRACTS,
    "CONTRACTS": DOC_TYPE_CONTRACTS,
    "CTR": DOC_TYPE_CONTRACTS,
    "ARI": DOC_TYPE_ASSET,
    "ASSET": DOC_TYPE_ASSET,
    "ASSETREG": DOC_TYPE_ASSET,
}

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
    "contract": DOC_TYPE_CONTRACTS,
    "contracts": DOC_TYPE_CONTRACTS,
    "agreement": DOC_TYPE_CONTRACTS,
    "agreements": DOC_TYPE_CONTRACTS,
    "contracten": DOC_TYPE_CONTRACTS,
    "overeenkomst": DOC_TYPE_CONTRACTS,
    "overeenkomsten": DOC_TYPE_CONTRACTS,
    "asset registration information": DOC_TYPE_ASSET,
    "asset registration": DOC_TYPE_ASSET,
    "activaregistratie": DOC_TYPE_ASSET,
    "unknown": DOC_TYPE_UNKNOWN,
    "unknown document type": DOC_TYPE_UNKNOWN,
    "onbekend": DOC_TYPE_UNKNOWN,
}

client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_KEY,
    api_version="2024-07-01-preview"
)
di_client = DocumentIntelligenceClient(DI_ENDPOINT, AzureKeyCredential(DI_KEY))
blob_service = BlobServiceClient.from_connection_string(CONN_STR)

QR_PATTERN = re.compile(r"^(\d{10,11})-([A-Za-z]{2})-([A-Za-z0-9]+)-(\d+)$")
QR_TEXT_SEARCH_PATTERN = re.compile(r"\d{10,11}-[A-Za-z]{2}-[A-Za-z0-9]+-\d+")


def cors(resp: func.HttpResponse):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
    return resp


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
    confidence = safe_number(result.get("classification_confidence"), 0)
    if document_type == DOC_TYPE_UNKNOWN:
        confidence = 0
    fields = result.get("fields") if isinstance(result.get("fields"), dict) else {}
    field_confidence = result.get("field_confidence") if isinstance(result.get("field_confidence"), dict) else {}
    return {
        "document_type": document_type,
        "classification_confidence": confidence,
        "contract_number": result.get("contract_number"),
        "region": result.get("region"),
        "fields": fields,
        "field_confidence": {str(k): safe_number(v, 0) for k, v in field_confidence.items()},
        "extraction_confidence": safe_number(result.get("extraction_confidence"), 0),
    }


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

    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported or unreadable image file")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Unable to convert uploaded image to PNG")
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
                data, _, _ = detector.detectAndDecode(candidate)
                if data:
                    return data.strip()
                retval, decoded_info, _, _ = detector.detectAndDecodeMulti(candidate)
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
    if not qr_value:
        return None
    match = QR_PATTERN.fullmatch(qr_value.strip())
    if not match:
        return None
    contract_number, region, doc_type_code, sequence_number = match.groups()
    code = doc_type_code.upper()
    mapped_type = QR_DOCUMENT_TYPE_MAP.get(code)
    if not mapped_type:
        return None
    return {
        "document_type": mapped_type,
        "classification_confidence": 99.5,
        "contract_number": contract_number,
        "region": region.upper(),
        "document_sequence": sequence_number,
        "qr_document_code": code,
    }


def analyze_with_document_intelligence(file_bytes):
    poller = di_client.begin_analyze_document(
        "prebuilt-layout", AnalyzeDocumentRequest(bytes_source=file_bytes)
    )
    result = poller.result()
    full_text = result.content or ""
    confidences = [
        word.confidence
        for page in (result.pages or [])
        for word in (page.words or [])
        if word.confidence is not None
    ]
    ocr_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0
    return full_text, round(ocr_confidence, 2)


def classify_extract_with_llm(text=None, images_bytes=None):
    prompt = """
You are a strict document classification and extraction engine. The document
may be in English, Dutch, or both.

Return exactly one document_type from this closed list:
1. Invoices
2. Certificate of Acceptance
3. Direct Debit Mandate
4. UBO Declarations
5. Contracts
6. Asset Registration Information
7. Unknown

Never invent or return another document type.

Classification guidance:
- Invoices: invoice, supplier/vendor invoice, billing or payment request,
  factuur, leveranciersfactuur, with invoice number, parties, line items,
  taxes, totals, or payment terms.
- Certificate of Acceptance: COA, acceptance certificate/confirmation,
  acceptatiecertificaat, bewijs van acceptatie, or acceptance of an asset,
  equipment, delivery, installation, or contractual item.
- Direct Debit Mandate: DDM, direct debit authorization, SEPA mandate,
  incassomachtiging, automatische incasso, or authority to debit an account.
- UBO Declarations: ultimate beneficial owner declaration, beneficial
  ownership statement, UBO-verklaring, or verklaring uiteindelijk
  belanghebbende.
- Contracts: contract, agreement, signed agreement, lease/service/purchase
  agreement, contract number, contract term, parties, governing law, payment
  terms, signatures, contracten, overeenkomst, or overeenkomsten.
- Asset Registration Information: asset/equipment/vehicle registration,
  asset information sheet, activaregistratie, bedrijfsmiddelregistratie,
  objectregistratie, or a document registering an asset and its identifiers.
- Unknown: the content does not clearly match one of the six business types.

A QR may be absent or unreadable. Classify from actual content and layout when
needed. Extract contract_number, region if clearly present as a 2-letter code,
important type-specific fields, per-field confidence, and overall extraction
confidence.

Return only valid JSON in this structure:
{
  "document_type": "Invoices | Certificate of Acceptance | Direct Debit Mandate | UBO Declarations | Contracts | Asset Registration Information | Unknown",
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
        content.append({"type": "text", "text": f"Document text:\n{text[:12000]}"})
    if images_bytes:
        for img in images_bytes:
            b64 = base64.b64encode(img).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    response = client.chat.completions.create(
        model=AOAI_DEPLOYMENT,
        messages=[{"role": "user", "content": content}],
        max_tokens=1500,
        temperature=0,
    )
    raw = strip_fences(response.choices[0].message.content)
    try:
        return validate_llm_result(json.loads(raw))
    except json.JSONDecodeError:
        logging.warning("LLM output was not valid JSON")
        return validate_llm_result({})


def process_document(filename, file_bytes):
    page_images = get_page_images(file_bytes, filename, dpi=200)
    qr_scan_images = get_page_images(file_bytes, filename, dpi=350)

    qr_value = decode_qr_from_pages(qr_scan_images)
    qr_source = "barcode_image" if qr_value else None

    full_text, ocr_confidence = analyze_with_document_intelligence(file_bytes)

    if not qr_value:
        text_match = find_qr_pattern_in_text(full_text)
        if text_match:
            qr_value = text_match
            qr_source = "text_fallback"

    approved_qr = parse_approved_qr(qr_value)

    if ocr_confidence >= OCR_TEXT_THRESHOLD:
        llm_result = classify_extract_with_llm(text=full_text)
        extraction_source = "document_intelligence_text"
    else:
        llm_result = classify_extract_with_llm(images_bytes=page_images)
        extraction_source = "azure_openai_vision_fallback"

    logging.info("Validated LLM result: %s", json.dumps(llm_result, ensure_ascii=False))

    extraction = {
        "fields": llm_result.get("fields", {}),
        "field_confidence": llm_result.get("field_confidence", {}),
        "overall_extraction_confidence": llm_result.get("extraction_confidence", 0),
        "extraction_source": extraction_source,
        "document_intelligence_ocr_confidence": ocr_confidence,
    }

    if approved_qr:
        classification = {
            "qr_identification": "Yes",
            "qr_raw_value": qr_value,
            "qr_source": qr_source,
            "qr_document_code": approved_qr["qr_document_code"],
            "document_sequence": approved_qr["document_sequence"],
            "classification_source": "approved_qr_mapping",
            "document_type": approved_qr["document_type"],
            "classification_confidence": approved_qr["classification_confidence"],
            "meets_threshold": True,
            "contract_number": approved_qr["contract_number"],
            "region": approved_qr["region"],
        }
    else:
        ai_type = normalize_document_type(llm_result.get("document_type"))
        ai_confidence = safe_number(llm_result.get("classification_confidence"), 0)
        meets_threshold = ai_type != DOC_TYPE_UNKNOWN and ai_confidence >= THRESHOLD
        classification = {
            "qr_identification": "Yes" if qr_value else "No",
            "qr_raw_value": qr_value,
            "qr_source": qr_source,
            "qr_approved_type": False if qr_value else None,
            "qr_pattern_valid": bool(qr_value and QR_PATTERN.fullmatch(qr_value.strip())) if qr_value else None,
            "classification_source": extraction_source,
            "document_type": ai_type if meets_threshold else DOC_TYPE_UNKNOWN,
            "classification_confidence": ai_confidence if meets_threshold else 0,
            "meets_threshold": meets_threshold,
            "contract_number": llm_result.get("contract_number"),
            "region": llm_result.get("region"),
        }

    return classification, extraction


@app.blob_trigger(
    arg_name="myblob", path="idp-docs-in/{name}",
    connection="AzureWebJobsStorage", source="EventGrid"
)
def DocumentClassifier(myblob: func.InputStream):
    # Start the timer when this function starts handling the Blob event.
    processing_start = time.perf_counter()

    filename = myblob.name.split("/")[-1]
    doc_id = filename.rsplit(".", 1)[0]
    logging.info("Processing document: %s", filename)

    try:
        file_bytes = myblob.read()
        classification, extraction = process_document(filename, file_bytes)
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

    # Elapsed time includes reading the input Blob, QR processing, OCR,
    # Azure OpenAI classification/extraction, and result construction.
    processing_seconds = round(time.perf_counter() - processing_start, 2)
    classification["Time taken to process the Document"] = f"{processing_seconds:.2f} seconds"

    blob_service.get_blob_client(CLASS_CONTAINER, f"{doc_id}.json").upload_blob(
        json.dumps(classification, indent=2, ensure_ascii=False), overwrite=True
    )
    blob_service.get_blob_client(EXTRACT_CONTAINER, f"{doc_id}.json").upload_blob(
        json.dumps(extraction, indent=2, ensure_ascii=False), overwrite=True
    )
    logging.info("Saved results for %s; processing time %.2f seconds", doc_id, processing_seconds)


@app.route(route="documents", methods=["GET"])
def list_documents(req: func.HttpRequest) -> func.HttpResponse:
    results = []
    for blob in blob_service.get_container_client(CLASS_CONTAINER).list_blobs():
        doc_id = blob.name.rsplit(".", 1)[0]
        try:
            raw = blob_service.get_blob_client(CLASS_CONTAINER, blob.name).download_blob().readall()
            classification = json.loads(raw)
        except Exception:
            classification = {}
        results.append({"id": doc_id, "name": blob.name, **classification})
    return cors(func.HttpResponse(json.dumps(results, ensure_ascii=False), mimetype="application/json"))


@app.route(route="documents/{id}", methods=["GET"])
def get_document(req: func.HttpRequest) -> func.HttpResponse:
    doc_id = req.route_params.get("id")
    try:
        raw = blob_service.get_blob_client(CLASS_CONTAINER, f"{doc_id}.json").download_blob().readall()
        classification = json.loads(raw)
    except Exception:
        return cors(func.HttpResponse(json.dumps({"error": "not found"}), status_code=404, mimetype="application/json"))
    try:
        raw = blob_service.get_blob_client(EXTRACT_CONTAINER, f"{doc_id}.json").download_blob().readall()
        extraction = json.loads(raw)
    except Exception:
        extraction = {"fields": {}, "field_confidence": {}, "overall_extraction_confidence": 0}
    return cors(func.HttpResponse(
        json.dumps({"classification": classification, "extraction": extraction}, ensure_ascii=False),
        mimetype="application/json"
    ))


@app.route(route="documents/{id}/file", methods=["GET"])
def get_document_file(req: func.HttpRequest) -> func.HttpResponse:
    doc_id = req.route_params.get("id")
    container = blob_service.get_container_client(IN_CONTAINER)
    for ext, content_type in [(".pdf", "application/pdf"), (".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg")]:
        blob_client = container.get_blob_client(f"{doc_id}{ext}")
        if blob_client.exists():
            return cors(func.HttpResponse(blob_client.download_blob().readall(), mimetype=content_type))
    return cors(func.HttpResponse(json.dumps({"error": "file not found"}), status_code=404, mimetype="application/json"))


@app.route(route="documents/upload", methods=["POST"])
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    file_bytes = req.get_body()
    if not file_bytes:
        return cors(func.HttpResponse(json.dumps({"error": "empty file"}), status_code=400, mimetype="application/json"))

    filename = os.path.basename(req.params.get("filename") or f"{uuid.uuid4()}.pdf")
    extension = os.path.splitext(filename)[1].lower()
    if extension not in {".pdf", ".png", ".jpg", ".jpeg"}:
        return cors(func.HttpResponse(
            json.dumps({"error": "unsupported file type; allowed types are PDF, PNG, JPG, and JPEG"}),
            status_code=400, mimetype="application/json"
        ))

    blob_service.get_blob_client(IN_CONTAINER, filename).upload_blob(file_bytes, overwrite=True)
    doc_id = filename.rsplit(".", 1)[0]
    return cors(func.HttpResponse(json.dumps({
        "status": "uploaded",
        "processing_status": "pending",
        "document_id": doc_id,
        "filename": filename
    }), mimetype="application/json"))
