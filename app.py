import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pypdf import PdfReader
import tempfile
import os
import requests
from dotenv import load_dotenv

app = FastAPI(title="AI Contract Analyzer")

# --------------------------
# CORS
# --------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()
# --------------------------
# CONFIGURATION
# --------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --------------------------
# HELPER FUNCTIONS 
# --------------------------
def validate_pdf(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

async def save_temp_pdf(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        return tmp.name

def extract_text(pdf_path: str):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text = full_text + "\n" + text
    return normalize_text(full_text)

def normalize_text(text: str):
    return " ".join(text.split())

def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)

# --------------------------
# CORE AGENT LOGIC
# --------------------------
def call_gemini(system_prompt: str, user_text: str):
    safe_text = user_text[:100000]
    
    combined_prompt = f"{system_prompt}\n\nData:\n{safe_text}"
    
    payload = {
        "contents": [{
            "parts": [{"text": combined_prompt}]
        }]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status() 
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------
# AGENT DEFINITIONS 
# --------------------------
def agent_the_offer(text: str):
    prompt = """
    Role: Compensation Expert.
    Task: Extract: 1. Compensation (Salary/Fee) 2. Payment Schedule 3. Job Title 4. Start Date.
    
    Format:
    <b>Job Title:</b> [Title] (Ref: [Section/Para])
    <b>Compensation:</b> [Amount] (Ref: [Section/Para])
    <b>Payment Schedule:</b> [Short Details] (Ref: [Section/Para])
    <b>Start Date:</b> [Date] (Ref: [Section/Para])
    
    Output must be in bulletpoints only.
    Constraint: Use HTML <b> tags for labels. Bulletpoints only. NO asterisks.
    MUST ADD A SPACE OF 1 EMPTY LINE After Each Item.ONLY 1 LINE BETWEEN EACH ITEM.
    Language Constraint: Use Simple easy english for better understanding.
    Output must be in bulletpoints only.
    """
    return call_gemini(prompt, text)

def agent_the_lifestyle(text: str):
    prompt = """
    Role: HR Specialist.
    Task: Extract: 1. Vacation/PTO 2. Benefits 3. IP Rights.
    
    Format:
    <b>Vacation:</b> [SHORT Details] (Ref: [Section/Para])
    <b>Benefits:</b> [SHORT Details] (Ref: [Section/Para])
    <b>IP Rights:</b> [SHORT Details] (Ref: [Section/Para])
    
    Constraint: Use HTML <b> tags for labels. Bulletpoints only.STRICTLY NO EMOJIS. NO asterisks.
    MUST ADD A SPACE OF 1 EMPTY LINE After Each Item.ONLY 1 LINE BETWEEN EACH ITEM.
    Language Constraint: Use Simple easy english for better understanding.
    Output must be in bulletpoints only.
    """
    return call_gemini(prompt, text)

def agent_the_lawyer(text: str):
    prompt = """
    Role: Defense Attorney.
    Task: You are a Defense Attorney. Find traps: 1. Termination 2. Non-Compete 3. Indemnification 4. Severance.
    Format:
    - <b>Clause [Number]:</b> [Risk Name] - [SHORT Explanation] (Ref: [Section/Para])
    - <b>Clause [Number]:</b> [Risk Name] - [SHORT Explanation] (Ref: [Section/Para])
    
    Constraint: Use HTML <b> tags for labels. List suspicious clauses.
    MUST ADD A SPACE OF 1 EMPTY LINE After Each Item.ONLY 1 LINE BETWEEN EACH ITEM.
    Language Constraint: Use Simple easy english for better understanding.
    Output must be in bulletpoints only.
    """
    return call_gemini(prompt, text)

def agent_the_executive(offer, lifestyle, lawyer):
    prompt = f"""
    Role: Career Coach.
    Task: Provide a final strategic summary based on these reports:
    
    [OFFER REPORT]
    {offer}
    
    [LIFESTYLE REPORT]
    {lifestyle}
    
    [LEGAL RISKS]
    {lawyer}
    
    Output Format:
    <b>Summary:</b> [2-3 sentence summary]
    
    Constraint: Use HTML <b> tags for labels. NO SCORING. NO EMOJIS.
    Language Constraint: Use Simple easy english for better understanding.
    """
    return call_gemini(prompt, "Summarize now.")
# --------------------------
# MAIN ENDPOINT
# --------------------------
@app.post("/analyze-contract")
async def analyze_contract(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None)
):
    if not file and not raw_text:
         raise HTTPException(status_code=400, detail="Provide PDF or Text")

    text = ""
    if file:
        validate_pdf(file)
        temp_path = await save_temp_pdf(file)
        try:
            text = extract_text(temp_path)
        finally:
            cleanup_file(temp_path)
    else:
        text = normalize_text(raw_text)

    print("--- Starting Deep Analysis ---")
    
    offer = agent_the_offer(text)
    print("Agent 1 (Offer) Done.")
    time.sleep(1)

    lifestyle = agent_the_lifestyle(text)
    print("Agent 2 (Lifestyle) Done.")
    time.sleep(10)

    lawyer = agent_the_lawyer(text)
    print("Agent 3 (Lawyer) Done.")
    time.sleep(1)

    verdict = agent_the_executive(offer, lifestyle, lawyer)
    print("Agent 4 (Verdict) Done.")

    return {
        "status": "success",
        "analysis": {
            "part_1_the_offer": offer,
            "part_2_lifestyle_and_ip": lifestyle,
            "part_3_legal_traps": lawyer,
            "part_4_executive_verdict": verdict
        }
    }