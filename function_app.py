import azure.functions as func
import logging, os, json, base64, uuid, re
import numpy as np
import cv2
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import fitz

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

client = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version="2024-07-01-preview")
di_client = DocumentIntelligenceClient(DI_ENDPOINT, AzureKeyCredential(DI_KEY))
blob_service = BlobServiceClient.from_connection_string(CONN_STR)

QR_PATTERN = re.compile(r'^(\d{10,11})-([A-Za-z]{2})-([A-Za-z0-9]+)-(\d+)$')
QR_TEXT_SEARCH_PATTERN = re.compile(r'\d{10,11}-[A-Za-z]{2}-[A-Za-z0-9]+-\d+')

def cors(resp: func.HttpResponse):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS"
    return resp

def strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*|```\s*$", "", text, flags=re.MULTILINE).strip()
    return text

def get_page_images(file_bytes, filename, dpi=200):
    if filename.lower().endswith(".pdf"):
        images = []
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(pix.tobytes("png"))
        return images
    return [file_bytes]

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
                data, points, _ = detector.detectAndDecode(candidate)
                if data:
                    return data
                retval, decoded_info, _, _ = detector.detectAndDecodeMulti(candidate)
                if retval:
                    for info in decoded_info:
                        if info:
                            return info
        except Exception as e:
            logging.warning(f"QR decode failed on a page: {e}")
    return None

def find_qr_pattern_in_text(text):
    if not text:
        return None
    match = QR_TEXT_SEARCH_PATTERN.search(text)
    return match.group(0) if match else None

def analyze_with_document_intelligence(file_bytes):
    poller = di_client.begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(bytes_source=file_bytes))
    result = poller.result()
    full_text = result.content or ""
    word_confidences = []
    for page in (result.pages or []):
        for word in (page.words or []):
            if word.confidence is not None:
                word_confidences.append(word.confidence)
    ocr_confidence = (sum(word_confidences) / len(word_confidences) * 100) if word_confidences else 0
    return full_text, round(ocr_confidence, 2)

def classify_extract_with_llm(text=None, images_bytes=None):
    prompt = (
        "You are a document classification and extraction engine. Documents may be in Dutch or English. "
        "Classify this document into a document_type. Known examples include: Invoice, "
        "Asset Registration Information, COA, DDM, UBO — but if none of these fit, invent a short, "
        "clear English type name that fits instead; do not force a bad match. "
        "Also extract any contract_number and region (2-letter country code) you can find, plus a "
        "'fields' object containing other key data points relevant to this document type. "
        "Return ONLY this JSON structure, no extra text: "
        '{"document_type": "...", "classification_confidence": 0-100, "contract_number": "..." or null, '
        '"region": "..." or null, "fields": {...}, "field_confidence": {"field_name": 0-100, ...}, '
        '"extraction_confidence": 0-100}'
    )
    content = [{"type": "text", "text": prompt}]
    if text:
        content.append({"type": "text", "text": f"Document text:\n{text[:12000]}"})
    if images_bytes:
        for img in images_bytes:
            b64 = base64.b64encode(img).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    response = client.chat.completions.create(
        model=AOAI_DEPLOYMENT, messages=[{"role": "user", "content": content}],
        max_tokens=1500, temperature=0,
    )
    raw = strip_fences(response.choices[0].message.content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logging.warning("LLM classification output wasn't valid JSON")
        return {"document_type": "Unknown", "classification_confidence": 0, "contract_number": None,
                "region": None, "fields": {}, "field_confidence": {}, "extraction_confidence": 0}

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

    classification_debug_text = full_text[-600:]

    if ocr_confidence >= 90:
        llm_result = classify_extract_with_llm(text=full_text)
        extraction_source = "document_intelligence_text"
    else:
        llm_result = classify_extract_with_llm(images_bytes=page_images)
        extraction_source = "azure_openai_vision_fallback"

    extraction = {
        "fields": llm_result.get("fields", {}),
        "field_confidence": llm_result.get("field_confidence", {}),
        "overall_extraction_confidence": llm_result.get("extraction_confidence", 0),
        "extraction_source": extraction_source,
        "document_intelligence_ocr_confidence": ocr_confidence
    }

    if qr_value:
        m = QR_PATTERN.match(qr_value.strip())
        if m:
            contract_number, region, doc_type_code, _seq = m.groups()
            doc_type, qr_conf = doc_type_code.upper(), 99.5
        else:
            contract_number, region, doc_type, qr_conf = None, None, "Unknown", 50.0

        classification = {
            "qr_identification": "Yes",
            "qr_raw_value": qr_value,
            "qr_source": qr_source,
            "document_type": doc_type if qr_conf >= THRESHOLD else "Unknown",
            "classification_confidence": qr_conf,
            "meets_threshold": qr_conf >= THRESHOLD,
            "contract_number": contract_number,
            "region": region
            #"debug_ocr_tail": classification_debug_text
        }
    else:
        conf = llm_result.get("classification_confidence", 0)
        classification = {
            "qr_identification": "No",
            "qr_raw_value": None,
            "qr_source": None,
            "document_type": llm_result.get("document_type", "Unknown") if conf >= THRESHOLD else "Unknown",
            "classification_confidence": conf,
            "meets_threshold": conf >= THRESHOLD,
            "contract_number": llm_result.get("contract_number"),
            "region": llm_result.get("region")
            #"debug_ocr_tail": classification_debug_text
        }

    return classification, extraction

@app.blob_trigger(arg_name="myblob", path="idp-docs-in/{name}", connection="AzureWebJobsStorage", source="EventGrid")
def DocumentClassifier(myblob: func.InputStream):
    filename = myblob.name.split("/")[-1]
    doc_id = filename.rsplit('.', 1)[0]
    logging.info(f"Processing document: {filename}")
    file_bytes = myblob.read()

    try:
        classification, extraction = process_document(filename, file_bytes)
    except Exception as e:
        logging.error(f"Processing failed for {filename}: {e}")
        classification = {"qr_identification": "No", "qr_raw_value": None, "qr_source": None, "document_type": "Unknown",
                           "classification_confidence": 0, "meets_threshold": False,
                           "contract_number": None, "region": None, "error": str(e)}
        extraction = {"fields": {}, "field_confidence": {}, "overall_extraction_confidence": 0}

    blob_service.get_blob_client(CLASS_CONTAINER, f"{doc_id}.json").upload_blob(json.dumps(classification, indent=2), overwrite=True)
    blob_service.get_blob_client(EXTRACT_CONTAINER, f"{doc_id}.json").upload_blob(json.dumps(extraction, indent=2), overwrite=True)
    logging.info(f"Saved classification + extraction for {doc_id}")

@app.route(route="documents", methods=["GET"])
def list_documents(req: func.HttpRequest) -> func.HttpResponse:
    results = []
    for blob in blob_service.get_container_client(CLASS_CONTAINER).list_blobs():
        doc_id = blob.name.rsplit('.', 1)[0]
        try:
            classification = json.loads(blob_service.get_blob_client(CLASS_CONTAINER, blob.name).download_blob().readall())
        except Exception:
            classification = {}
        results.append({"id": doc_id, "name": blob.name, **classification})
    return cors(func.HttpResponse(json.dumps(results), mimetype="application/json"))

@app.route(route="documents/{id}", methods=["GET"])
def get_document(req: func.HttpRequest) -> func.HttpResponse:
    doc_id = req.route_params.get("id")
    try:
        classification = json.loads(blob_service.get_blob_client(CLASS_CONTAINER, f"{doc_id}.json").download_blob().readall())
    except Exception:
        return cors(func.HttpResponse(json.dumps({"error": "not found"}), status_code=404, mimetype="application/json"))
    try:
        extraction = json.loads(blob_service.get_blob_client(EXTRACT_CONTAINER, f"{doc_id}.json").download_blob().readall())
    except Exception:
        extraction = {"fields": {}, "field_confidence": {}, "overall_extraction_confidence": 0}
    return cors(func.HttpResponse(json.dumps({"classification": classification, "extraction": extraction}), mimetype="application/json"))

@app.route(route="documents/{id}/file", methods=["GET"])
def get_document_file(req: func.HttpRequest) -> func.HttpResponse:
    doc_id = req.route_params.get("id")
    container_client = blob_service.get_container_client(IN_CONTAINER)
    for ext, content_type in [(".pdf", "application/pdf"), (".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg")]:
        blob_client = container_client.get_blob_client(f"{doc_id}{ext}")
        if blob_client.exists():
            return cors(func.HttpResponse(blob_client.download_blob().readall(), mimetype=content_type))
    return cors(func.HttpResponse(json.dumps({"error": "file not found"}), status_code=404, mimetype="application/json"))

@app.route(route="documents/upload", methods=["POST"])
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    file_bytes = req.get_body()
    filename = req.params.get("filename", f"{uuid.uuid4()}.pdf")
    blob_service.get_blob_client(IN_CONTAINER, filename).upload_blob(file_bytes, overwrite=True)
    return cors(func.HttpResponse(json.dumps({"status": "uploaded", "filename": filename}), mimetype="application/json"))