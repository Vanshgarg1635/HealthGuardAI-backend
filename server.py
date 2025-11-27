# backend.py
import os
import io
import re
import json
import uuid
import logging
import pytesseract
import cloudinary
import cloudinary.uploader

from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
import jwt

from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# LangChain / Vector store imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# Pydantic
from pydantic import BaseModel, Field, EmailStr, ConfigDict

# Setup
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthguard-backend")

# MongoDB setup
MONGO_URL = os.environ["MONGODB_URI"]
client = AsyncIOMotorClient(MONGO_URL)
db = client[os.environ["DB_NAME"]]

# Cloudinary config (kept)
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
)

# LLM + embeddings (Gemini)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GEMINI_API_KEY)

# Security / JWT
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.environ.get("JWT_EXPIRATION_HOURS", 24))
security = HTTPBearer()

# Global retriever (populated at startup)
_global_retriever = None

# ---------- Pydantic models ----------
class UserSignup(BaseModel):
    email: EmailStr
    username: str
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    username: str
    unique_token: str
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: User

class Report(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    user_id: str
    report_date: str
    original_files: List[str]
    result_pdf: str
    extracted_data: Dict[str, Any]
    created_at: datetime

class FamilyInvite(BaseModel):
    invitee_token: str

class FamilyLink(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    requester_id: str
    requester_username: str
    invitee_id: str
    invitee_username: str
    status: str
    created_at: datetime
    accepted_at: Optional[datetime] = None

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# ---------- Utils ----------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user = await db.users.find_one({"id": user_id}, {"_id": 0})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        if isinstance(user["created_at"], str):
            user["created_at"] = datetime.fromisoformat(user["created_at"])
        return User(**user)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.exception("Auth error: %s", e)
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# ---------- OCR + JSON extraction helpers ----------
def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Robust OCR extraction for images and PDFs.
    Returns extracted text (possibly empty string on failure).
    """
    try:
        logger.info("Starting OCR extraction for file: %s", filename)
        lower = filename.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            image = Image.open(io.BytesIO(file_bytes))
            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Improve contrast
            image = ImageOps.autocontrast(image)
            # Perform OCR
            text = pytesseract.image_to_string(image)
            logger.info("OCR image length: %d", len(text))
            return text or ""
        elif lower.endswith(".pdf"):
            # Convert PDF to images with decent DPI
            images = convert_from_bytes(file_bytes, dpi=300, fmt="png")
            if not images:
                logger.warning("PDF conversion produced 0 images.")
                return ""
            parts = []
            for i, image in enumerate(images):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = ImageOps.autocontrast(image)
                page_text = pytesseract.image_to_string(image)
                parts.append(page_text or "")
            full = "\n\n".join(parts)
            logger.info("OCR pdf length: %d", len(full))
            return full or ""
        else:
            logger.warning("Unsupported file format for OCR: %s", filename)
            return ""
    except Exception as e:
        logger.exception("Error in OCR for %s: %s", filename, e)
        return ""

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Try multiple strategies to extract JSON from text:
    1) If the entire text is JSON
    2) JSON in code fences (```json ... ``` or ``` ... ```)
    3) Largest {...} substring heuristic
    """
    if not text:
        return None
    # 1) direct JSON
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # 2) find fenced blocks
    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for fm in fence_matches:
        try:
            return json.loads(fm)
        except Exception:
            continue
    # 3) fallback: find balanced braces and try largest chunks
    braces = []
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            stack.append(i)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start:i+1]
                    braces.append(candidate)
                    start = None
    for cand in sorted(braces, key=len, reverse=True):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

# ---------- RAG + LLM analysis ----------
# JSON schema instruction to enforce structure
JSON_SCHEMA_INSTRUCTIONS = """
Return ONLY a JSON object EXACTLY matching the schema below. If a field is not present, use null or empty lists/strings.
Schema:
{
  "patient": {
    "name": "string or null",
    "age": "number or null",
    "gender": "string or null",
    "height": "string or null",
    "weight": "string or null"
  },
  "parameters": [
    {
      "name": "string",
      "value": "string",
      "units": "string or null",
      "normal_range": "string or null",
      "severity": "Normal | Low | High | Critical",
      "color": "green|yellow|orange|red",
      "recommendation": "string or null"
    }
  ],
  "specialist_recommendations": ["list of strings"],
  "diet_exercise_plan": "string or null",
  "raw_analysis_text": "string"
}
Do not output any extra commentary or stray text. Respond only with JSON.
"""

async def analyze_health_parameters(extracted_text: str, retriever=None) -> Dict[str, Any]:
    """
    Analyze extracted_text using LLM + optional retriever.
    Always returns a strict JSON structure.
    """
    if not extracted_text or extracted_text.strip() == "":
        return {
            "error": "No text extracted",
            "parameters": [],
            "raw_analysis_text": ""
        }

    # ---------------------------------------------------------------------
    # OPTIONAL RETRIEVAL HELP (but LLM should NOT depend on it)
    # ---------------------------------------------------------------------
    retrieval_context = ""
    if retriever:
        try:
            docs = retriever.invoke(extracted_text)   # modern LC call
            top_texts = [
                d.page_content
                for d in docs[:5]
                if hasattr(d, "page_content") and d.page_content
            ]
            if top_texts:
                retrieval_context = "\n\n".join(top_texts)
        except Exception as e:
            logger.warning("Retriever failed (ignoring and continuing): %s", e)

    # ---------------------------------------------------------------------
    # BUILD SUPER CLEAR PROMPT
    # ---------------------------------------------------------------------
    final_prompt = f"""
You are an advanced medical AI system.

Your tasks:

1. **Extract ALL measurable health parameters** from the medical report text.
2. **Interpret each parameter** using BOTH:
   - your own medical knowledge (normal ranges, clinical meaning, risks),
   - and the retrieval context IF available.
3. For each parameter, assign a severity level:
   - "Normal"
   - "Low"
   - "High"
   - "Abnormal"
   - "Critical"

4. **Write detailed medical explanations**, not short phrases.

5. **SPECIALIST RECOMMENDATIONS → MUST BE VERY DETAILED**
   For each specialist you recommend:
   - Specify the exact type of doctor (e.g., Nephrologist, Cardiologist, Endocrinologist).
   - Explain *why* this specialist is relevant.
   - List **specific diagnostic tests** that the patient should get.
   - Mention potential **diseases or conditions to rule out**.
   - Minimum 2–4 sentences per specialist.

6. **DIET & EXERCISE PLAN → VERY DETAILED**
   Your plan must include:
   - A section for **Diet Recommendations** (bullet points)
   - A section for **Exercise Plan** (frequency, type, intensity)(bullet points)
   - A section for **Lifestyle Advice** (stress, sleep, hydration, habits)(bullet points)
   - Tie all recommendations to the patient’s abnormal parameters.
   -These sections must be concise and to the point and short with bullet points.
   -Each section must have a maximum of 5 bullet points.

7. Your response MUST be **strict, valid JSON ONLY**.
   - No markdown
   - No commentary
   - No text outside the JSON
   - No trailing commas

Required JSON schema:
{JSON_SCHEMA_INSTRUCTIONS}

If the report text is unclear or missing, return:
{{
  "patient": {{"name": null, "age": null, "gender": null, "height": null, "weight": null}},
  "parameters": [],
  "specialist_recommendations": [],
  "diet_exercise_plan": null,
  "raw_analysis_text": "<your reasoning here>"
}}

----------------------
RETRIEVAL CONTEXT (optional):
{retrieval_context or "[No retrieval data available — rely on your own medical expertise]"}
----------------------

MEDICAL REPORT TEXT:
{extracted_text}
----------------------

Now return ONLY the final JSON. Nothing else.
"""


    logger.debug("LLM prompt length: %d", len(final_prompt))

    # ---------------------------------------------------------------------
    # CALL LLM SAFELY
    # ---------------------------------------------------------------------
    try:
        response = await llm.ainvoke(final_prompt)
        raw_text = getattr(response, "content", None) or str(response)
        logger.info("LLM raw response length: %d", len(raw_text))

        # Save raw text for debugging
        try:
            await db.llm_raw_responses.insert_one({
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_snippet": final_prompt[:2000],
                "response": raw_text
            })
        except Exception as e:
            logger.warning("Could not save raw LLM response: %s", e)

        # SAFE JSON EXTRACTION
        parsed = extract_json_from_text(raw_text)

        if parsed:
            # Guarantee required keys exist
            parsed.setdefault("patient", {"name": None, "age": None, "gender": None, "height": None, "weight": None})
            parsed.setdefault("parameters", [])
            parsed.setdefault("specialist_recommendations", [])
            parsed.setdefault("diet_exercise_plan", None)
            parsed.setdefault("raw_analysis_text", raw_text[:10000])
            return parsed

        # If JSON parsing failed → return fallback structure
        logger.warning("LLM returned NON-JSON, sending fallback.")
        return {
            "error": "LLM response not JSON",
            "parameters": [],
            "raw_analysis_text": raw_text
        }

    except Exception as e:
        logger.exception("LLM invocation failed: %s", e)
        return {
            "error": f"LLM error: {e}",
            "parameters": [],
            "raw_analysis_text": ""
        }


# ---------- PDF generation ----------
def generate_pdf_report(analysis_data: Dict[str, Any], username: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1e40af"),
        spaceAfter=12,
        alignment=1,
    )

    section_title = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    bullet_style = styles["Bullet"]
    normal_style = styles["Normal"]

    # -------------------------------------------------------
    # Utility: Parse specialist text into structured dict
    # -------------------------------------------------------
    def parse_specialist_text(text: str):
        sections = []
        parts = text.split(".,")  # split by sentence blocks

        for p in parts:
            p = p.strip()
            if not p:
                continue

            if ":" in p:
                title, desc = p.split(":", 1)
                title = title.strip()
                desc = desc.strip()

                # break desc into bullet points
                bullets = [x.strip() for x in desc.split(".") if x.strip()]

                sections.append({
                    "title": title,
                    "details": bullets
                })
        return sections

    # -------------------------------------------------------
    # HEADER
    # -------------------------------------------------------
    story.append(Paragraph("HealthGuard AI - Medical Report Analysis", title_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(f"<b>Patient:</b> {username or 'Unknown'}", normal_style))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", normal_style))
    story.append(Spacer(1, 0.2 * inch))

    # -------------------------------------------------------
    # RAW ANALYSIS
    # -------------------------------------------------------
    if analysis_data.get("raw_analysis_text"):
        story.append(Paragraph("📄 <b>Raw Analysis</b>", section_title))
        story.append(Paragraph(
            analysis_data["raw_analysis_text"].replace("\n", "<br/>")[:4000],
            normal_style
        ))
        story.append(Spacer(1, 0.2 * inch))

    # -------------------------------------------------------
    # PARAMETERS
    # -------------------------------------------------------
    params = analysis_data.get("parameters") or []
    story.append(Paragraph("📊 <b>Health Parameters</b>", section_title))

    for p in params:
        if isinstance(p, dict):
            name = p.get("name", "N/A")
            value = p.get("value", "N/A")
            nr = p.get("normal_range", "N/A")
            severity = p.get("severity", "N/A")
            rec = p.get("recommendation", "")

            story.append(Paragraph(f"<b>{name}</b>: {value} (Normal: {nr})", normal_style))
            story.append(Paragraph(f"• <b>Severity:</b> {severity}", bullet_style))

            if rec:
                story.append(Paragraph(f"• <b>Recommendation:</b> {rec}", bullet_style))

            story.append(Spacer(1, 0.12 * inch))

    # -------------------------------------------------------
    # -------------------------------------------------------
    # SPECIALIST RECOMMENDATIONS — ALWAYS STRUCTURED
    # -------------------------------------------------------
    story.append(Paragraph("🩺 <b>Specialist Recommendations</b>", section_title))

    specialists_raw = analysis_data.get("specialist_recommendations", [])

    # Normalized list
    structured_specs = []

    # Force everything into structured format
    def convert_to_struct(item):
        if isinstance(item, dict):
            # Ensure keys exist
            return {
                "title": item.get("title", "Specialist"),
                "details": item.get("details", [])
            }

        if isinstance(item, str):
            # Convert messy text → structured dict
            if ":" in item:
                title, desc = item.split(":", 1)
                bullets = [
                    x.strip().rstrip(".,")
                    for x in desc.split(".")
                    if x.strip() and len(x.strip()) > 2
                ]
                return {
                    "title": title.strip(),
                    "details": bullets
                }

            # If string with no ":" → fallback
            return {
                "title": "Specialist",
                "details": [item.strip()]
            }

        # Unknown type fallback
        return {
            "title": "Specialist",
            "details": [str(item)]
        }

    # Normalize all items
    if isinstance(specialists_raw, list):
        structured_specs = [convert_to_struct(x) for x in specialists_raw]
    elif isinstance(specialists_raw, str):
        structured_specs = [convert_to_struct(specialists_raw)]
    else:
        structured_specs = []

    # Render specialists
    for sp in structured_specs:
        story.append(Paragraph(f"• <b>{sp['title']}</b>", bullet_style))
        for d in sp["details"]:
            story.append(Paragraph(f"&nbsp;&nbsp;– {d}", normal_style))
        story.append(Spacer(1, 0.1 * inch))

    # -------------------------------------------------------
    # DIET & EXERCISE PLAN — AUTO STRUCTURED
    # -------------------------------------------------------
    story.append(Paragraph("🥗 <b>Diet & Lifestyle Plan</b>", section_title))

    plan_raw = analysis_data.get("diet_exercise_plan")

    diet_list, exercise_list, lifestyle_list = [], [], []

    if isinstance(plan_raw, dict):
        diet_list = plan_raw.get("diet", []) or []
        exercise_list = plan_raw.get("exercise", []) or []
        lifestyle_list = plan_raw.get("lifestyle", []) or []

    elif isinstance(plan_raw, str):
        text = plan_raw.lower()

        def extract(section):
            if section in text:
                segment = text.split(section)[1]
                segment = segment.split("**")[0]
                return [
                    s.strip(" *.-").capitalize()
                    for s in segment.split("*")
                    if len(s.strip()) > 3
                ]
            return []

        diet_list = extract("diet")
        exercise_list = extract("exercise")
        lifestyle_list = extract("lifestyle")

    # Render Diet
    if diet_list:
        story.append(Paragraph("<b>🍏 Diet Recommendations</b>", normal_style))
        for d in diet_list:
            story.append(Paragraph(f"• {d}", bullet_style))
        story.append(Spacer(1, 0.1 * inch))

    # Render Exercise
    if exercise_list:
        story.append(Paragraph("<b>💪 Exercise Plan</b>", normal_style))
        for e in exercise_list:
            story.append(Paragraph(f"• {e}", bullet_style))
        story.append(Spacer(1, 0.1 * inch))

    # Render Lifestyle
    if lifestyle_list:
        story.append(Paragraph("<b>🌿 Lifestyle Advice</b>", normal_style))
        for l in lifestyle_list:
            story.append(Paragraph(f"• {l}", bullet_style))
        story.append(Spacer(1, 0.1 * inch))


    # -------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(
        "<i>Disclaimer: This report is generated by AI and should not replace professional medical advice.</i>",
        normal_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ---------- FAISS initialization ----------
async def initialize_health_standards():
    """
    Seed standard health parameter texts to DB (if empty) and build FAISS retriever.
    Returns a retriever object and sets _global_retriever.
    """
    global _global_retriever
    standard_data = [
        "Normal hemoglobin levels: Male 13.5-17.5 g/dL, Female 12.0-15.5 g/dL",
        "Normal RBC count: Male 4.5-5.9 million/mcL, Female 4.0-5.2 million/mcL",
        "Normal WBC count: 4,500-11,000 cells/mcL",
        "Normal platelet count: 150,000-450,000 per mcL",
        "Normal cholesterol: Less than 200 mg/dL",
        "Normal HDL: Greater than 40 mg/dL for men, greater than 50 mg/dL for women",
        "Normal LDL: Less than 100 mg/dL",
        "Normal triglycerides: Less than 150 mg/dL",
        "Normal creatinine: 0.7-1.3 mg/dL for men, 0.6-1.1 mg/dL for women",
        "Normal ALT: 7-56 units/L",
        "Normal AST: 10-40 units/L",
        "Normal vitamin D: 20-50 ng/mL",
        "Normal vitamin B12: 200-900 pg/mL",
    ]

    try:
        existing = await db.health_standards.find_one({})
        if not existing:
            doc = {"id": str(uuid.uuid4()), "standards": standard_data, "created_at": datetime.now(timezone.utc).isoformat()}
            await db.health_standards.insert_one(doc)
            logger.info("Seeded health standards into DB.")
        else:
            # If we have a DB copy, use that list if it's longer
            if existing.get("standards"):
                standard_data = existing["standards"]
    except Exception as e:
        logger.warning("Could not seed or read health_standards collection: %s", e)

    try:
        # Build FAISS index using the embeddings instance you already have
        faiss_index = FAISS.from_texts(standard_data, embeddings)
        retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        _global_retriever = retriever
        logger.info("Built FAISS retriever for health standards.")
        return retriever
    except Exception as e:
        logger.exception("Failed to build FAISS retriever: %s", e)
        _global_retriever = None
        return None

# ---------- FastAPI app and routes ----------
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Auth endpoints (unchanged behavior)
@api_router.post("/auth/signup", response_model=TokenResponse)
async def signup(user_data: UserSignup):
    if user_data.password != user_data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    existing_user = await db.users.find_one({"$or": [{"email": user_data.email}, {"username": user_data.username}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email or username already registered")
    user_id = str(uuid.uuid4())
    unique_token = str(uuid.uuid4())[:8].upper()
    hashed_pwd = hash_password(user_data.password)
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "username": user_data.username,
        "password_hash": hashed_pwd,
        "unique_token": unique_token,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.users.insert_one(user_doc)
    access_token = create_access_token(data={"sub": user_id})
    user_obj = User(id=user_id, email=user_data.email, username=user_data.username, unique_token=unique_token, created_at=datetime.now(timezone.utc))
    return TokenResponse(access_token=access_token, token_type="bearer", user=user_obj)

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    user = await db.users.find_one({"username": credentials.username})
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": user["id"]})
    user_obj = User(id=user["id"], email=user["email"], username=user["username"], unique_token=user["unique_token"], created_at=datetime.fromisoformat(user["created_at"]))
    return TokenResponse(access_token=access_token, token_type="bearer", user=user_obj)

# Report upload endpoint - improved pipeline
@api_router.post("/reports/upload", response_model=Report)
async def upload_reports(files: List[UploadFile] = File(...), current_user: User = Depends(get_current_user)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

    uploaded_urls = []
    all_extracted_text = ""

    # Process each file
    for file in files:
        file_bytes = await file.read()
        if len(file_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 20MB limit")

        # Upload original file to Cloudinary (optional: you can skip to save bandwidth)
        try:
            upload_result = cloudinary.uploader.upload(file_bytes, folder="healthguard/reports", resource_type="auto")
            uploaded_urls.append(upload_result.get("secure_url"))
        except Exception as e:
            logger.warning("Cloudinary upload of original file failed (continuing): %s", e)
            # Do not fail here — we can still OCR & analyze

        # Extract text
        extracted_text = extract_text_from_file(file_bytes, file.filename)
        logger.debug("Extracted text snippet: %s", (extracted_text or "")[:300])
        all_extracted_text += (extracted_text or "") + "\n\n"

    if not all_extracted_text.strip():
        logger.warning("No text extracted from any uploaded files.")
        analysis_data = {"error": "No text could be extracted from uploaded files.", "parameters": [], "raw_analysis_text": ""}
    else:
        # Perform analysis with RAG retriever
        analysis_data = await analyze_health_parameters(all_extracted_text, retriever=_global_retriever)

    # Ensure analysis_data is a dict
    if not isinstance(analysis_data, dict):
        analysis_data = {"error": "Unexpected analysis output", "parameters": [], "raw_analysis_text": str(analysis_data)}

    # Generate PDF (always returns something)
    try:
        pdf_bytes = generate_pdf_report(analysis_data, current_user.username)
    except Exception as e:
        logger.exception("PDF generation failed: %s", e)
        # Return a small fallback PDF containing the error
        fallback_text = {"error": "PDF generation failed", "details": str(e)}
        pdf_bytes = generate_pdf_report({"raw_analysis_text": json.dumps(fallback_text)}, current_user.username)

    # Upload PDF to Cloudinary
    try:
        pdf_upload = cloudinary.uploader.upload(io.BytesIO(pdf_bytes), folder="healthguard/results", resource_type="raw", format="pdf")
        pdf_url = pdf_upload.get("secure_url")
    except Exception as e:
        logger.exception("Cloudinary upload for PDF failed: %s", e)
        # If upload to Cloudinary fails, store PDF as binary in DB (not ideal for production)
        pdf_url = None
        try:
            storage_doc = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(), "pdf_bytes": pdf_bytes}
            await db.pdf_storage.insert_one(storage_doc)
            pdf_url = f"db://pdf_storage/{storage_doc['id']}"
        except Exception as e2:
            logger.exception("Failed to store PDF in DB fallback: %s", e2)

    # Save report into DB (include raw LLM response if present)
    report_id = str(uuid.uuid4())
    report_doc = {
        "id": report_id,
        "user_id": current_user.id,
        "report_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "original_files": uploaded_urls,
        "result_pdf": pdf_url,
        "extracted_data": analysis_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    try:
        await db.reports.insert_one(report_doc)
    except Exception as e:
        logger.exception("Failed to save report to DB: %s", e)

    return Report(
        id=report_id,
        user_id=current_user.id,
        report_date=report_doc["report_date"],
        original_files=uploaded_urls,
        result_pdf=pdf_url,
        extracted_data=analysis_data,
        created_at=datetime.now(timezone.utc),
    )

# Other endpoints unchanged (family, reports retrieval, chat etc.) — kept for parity with your original server
@api_router.get("/reports/user", response_model=List[Report])
async def get_user_reports(date: Optional[str] = None, current_user: User = Depends(get_current_user)):
    query = {"user_id": current_user.id}
    if date:
        query["report_date"] = date
    reports = await db.reports.find(query, {"_id": 0}).sort("created_at", -1).to_list(100)
    for report in reports:
        if isinstance(report.get("created_at"), str):
            report["created_at"] = datetime.fromisoformat(report["created_at"])
    return reports

@api_router.get("/reports/family", response_model=List[Report])
async def get_family_reports(current_user: User = Depends(get_current_user)):
    family_links = await db.family_links.find({
        "$or": [
            {"requester_id": current_user.id, "status": "accepted"},
            {"invitee_id": current_user.id, "status": "accepted"}
        ]
    }, {"_id": 0}).to_list(100)
    family_ids = []
    for link in family_links:
        if link["requester_id"] == current_user.id:
            family_ids.append(link["invitee_id"])
        else:
            family_ids.append(link["requester_id"])
    if not family_ids:
        return []
    reports = await db.reports.find({"user_id": {"$in": family_ids}}, {"_id": 0}).sort("created_at", -1).to_list(100)
    for report in reports:
        if isinstance(report.get("created_at"), str):
            report["created_at"] = datetime.fromisoformat(report["created_at"])
    return reports

@api_router.post("/family/invite")
async def invite_family_member(invite: FamilyInvite, current_user: User = Depends(get_current_user)):
    invitee = await db.users.find_one({"unique_token": invite.invitee_token})
    if not invitee:
        raise HTTPException(status_code=404, detail="User with this token not found")
    if invitee["id"] == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot invite yourself")
    existing = await db.family_links.find_one({
        "$or": [
            {"requester_id": current_user.id, "invitee_id": invitee["id"]},
            {"requester_id": invitee["id"], "invitee_id": current_user.id}
        ]
    })
    if existing:
        raise HTTPException(status_code=400, detail="Family link already exists")
    link_doc = {
        "id": str(uuid.uuid4()),
        "requester_id": current_user.id,
        "requester_username": current_user.username,
        "invitee_id": invitee["id"],
        "invitee_username": invitee["username"],
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accepted_at": None
    }
    await db.family_links.insert_one(link_doc)
    return {"message": f"Invite sent to {invitee['username']}", "status": "pending"}

@api_router.get("/family/invites", response_model=List[FamilyLink])
async def get_family_invites(current_user: User = Depends(get_current_user)):
    invites = await db.family_links.find({"invitee_id": current_user.id, "status": "pending"}, {"_id": 0}).to_list(100)
    for invite in invites:
        if isinstance(invite.get("created_at"), str):
            invite["created_at"] = datetime.fromisoformat(invite["created_at"])
    return invites

@api_router.post("/family/accept/{invite_id}")
async def accept_family_invite(invite_id: str, current_user: User = Depends(get_current_user)):
    invite = await db.family_links.find_one({"id": invite_id, "invitee_id": current_user.id})
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    await db.family_links.update_one({"id": invite_id}, {"$set": {"status": "accepted", "accepted_at": datetime.now(timezone.utc).isoformat()}})
    return {"message": "Invite accepted", "status": "accepted"}

@api_router.post("/family/reject/{invite_id}")
async def reject_family_invite(invite_id: str, current_user: User = Depends(get_current_user)):
    invite = await db.family_links.find_one({"id": invite_id, "invitee_id": current_user.id})
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    await db.family_links.delete_one({"id": invite_id})
    return {"message": "Invite rejected"}

@api_router.get("/family/members")
async def get_family_members(current_user: User = Depends(get_current_user)):
    family_links = await db.family_links.find({
        "$or": [
            {"requester_id": current_user.id, "status": "accepted"},
            {"invitee_id": current_user.id, "status": "accepted"}
        ]
    }, {"_id": 0}).to_list(100)
    members = []
    for link in family_links:
        if link["requester_id"] == current_user.id:
            members.append({"id": link["invitee_id"], "username": link["invitee_username"]})
        else:
            members.append({"id": link["requester_id"], "username": link["requester_username"]})
    return {"members": members}

@app.post("/family/remove/{family_link_id}")
async def remove_family_member(family_link_id: str, current_user: User = Depends(get_current_user)):
    """
    Remove a family member connection using the family link id.
    """
    # Find the link by family link id
    link = await db.family_links.find_one({"id": family_link_id})
    
    if not link:
        raise HTTPException(status_code=404, detail="Family link not found")
    
    # Ensure the current user is either the requester or the invitee
    if link["requester_id"] != current_user.id and link["invitee_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="You are not authorized to remove this family member")

    # Delete the family link
    await db.family_links.delete_one({"id": family_link_id})

    return {"message": f"Family member with link id {family_link_id} has been removed."}


@api_router.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, current_user: User = Depends(get_current_user)):
    # Fetch latest reports for context
    reports = await db.reports.find({"user_id": current_user.id}, {"_id": 0}).sort("created_at", -1).limit(3).to_list(3)
    context = "User recent health reports:\n"
    for report in reports:
        context += f"Date: {report['report_date']}\n"
        analysis_text = report.get("extracted_data", {}).get("raw_analysis_text") or report.get("extracted_data", {}).get("analysis")
        if analysis_text:
            context += f"Analysis snippet: {analysis_text[:500]}...\n\n"
    prompt = f"""{context}

User question: {message.message}

Provide a helpful response based on the health data. Always remind users to consult healthcare professionals for medical advice.

Respond in friendly tone.
"""
    try:
        response = await llm.ainvoke(prompt)
        raw = getattr(response, "content", None) or str(response)
        return ChatResponse(response=raw)
    except Exception as e:
        logger.exception("Chat error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process chat message")

# Startup / shutdown
@app.on_event("startup")
async def startup_event():
    await initialize_health_standards()
    logger.info("Health standards initialized and retriever built (if possible)")

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()