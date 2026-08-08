import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
import gridfs
import uvicorn

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "rsmssb_mcq")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing.")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = os.getenv("GEMINI_MODEL")

mongo = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = mongo[MONGODB_DB]
studies = db["studies"]
attempts = db["attempts"]
fs = gridfs.GridFS(db)

studies.create_index([("created_at", DESCENDING)])

app = FastAPI(title="RSMSSB Computer Instructor MCQ Platform", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=20)
    count: int = Field(default=50, ge=1, le=100)
    topic: str = ""

class DoubtRequest(BaseModel):
    question: str
    options: Dict[str, str]
    correctAnswer: str
    studentAnswer: str = ""
    explanation: str = ""
    doubt: str
    history: List[Dict[str, str]] = []

class SaveAttemptRequest(BaseModel):
    study_id: str
    correct: int
    wrong: int
    skipped: int
    accuracy: int
    answers: List[Dict[str, Any]] = []

QUESTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "topics": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        },
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "options": {
                        "type": "OBJECT",
                        "properties": {
                            "A": {"type": "STRING"},
                            "B": {"type": "STRING"},
                            "C": {"type": "STRING"},
                            "D": {"type": "STRING"}
                        },
                        "required": ["A", "B", "C", "D"]
                    },
                    "correctAnswer": {
                        "type": "STRING",
                        "enum": ["A", "B", "C", "D"]
                    },
                    "explanation": {"type": "STRING"},
                    "topic": {"type": "STRING"}
                },
                "required": [
                    "question", "options", "correctAnswer",
                    "explanation", "topic"
                ]
            }
        }
    },
    "required": ["topics", "questions"]
}

def trim_document(text: str, max_chars: int = 90000) -> str:
    if len(text) <= max_chars:
        return text
    third = max_chars // 3
    middle = len(text) // 2
    return (
        text[:third]
        + "\n\n[...middle section omitted...]\n\n"
        + text[middle-third//2:middle+third//2]
        + "\n\n[...later section...]\n\n"
        + text[-third:]
    )

def normalize_topic(value: str) -> str:
    return " ".join(str(value or "").strip().split())

def validate_generated(data: Any, requested_count: int):
    if not isinstance(data, dict):
        raise ValueError("Gemini did not return the expected object.")
    raw_topics = data.get("topics", [])
    raw_questions = data.get("questions", [])
    topics = []
    for t in raw_topics:
        t = normalize_topic(t)
        if t and t.lower() not in {x.lower() for x in topics}:
            topics.append(t)

    valid = []
    seen = set()
    topic_lookup = {t.lower(): t for t in topics}

    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        options = item.get("options", {})
        correct = item.get("correctAnswer", "")
        explanation = str(item.get("explanation", "")).strip()
        topic = normalize_topic(item.get("topic", ""))

        if not q or not isinstance(options, dict):
            continue
        if correct not in ["A", "B", "C", "D"]:
            continue
        opts = {k: str(options.get(k, "")).strip() for k in ["A","B","C","D"]}
        if any(not opts[k] for k in opts):
            continue
        if not topic:
            continue

        canonical = topic_lookup.get(topic.lower())
        if canonical is None:
            topics.append(topic)
            topic_lookup[topic.lower()] = topic
            canonical = topic

        key = q.casefold()
        if key in seen:
            continue
        seen.add(key)

        valid.append({
            "question": q,
            "options": opts,
            "correctAnswer": correct,
            "explanation": explanation,
            "topic": canonical
        })
        if len(valid) >= requested_count:
            break

    if not valid:
        raise ValueError("Gemini did not generate any valid questions.")
    return topics, valid

def generate_questions(document_text: str, count: int, focus_topic: str = ""):
    focus = ""
    if focus_topic.strip():
        focus = f"""
FOCUS TOPIC:
{focus_topic.strip()}
Generate questions only about this topic. If it is not supported by the source,
return no questions.
"""

    system = f"""
You are an expert question writer for the RSMSSB Computer Instructor examination.

Create up to {count} original MCQs strictly from the supplied study material.

FIRST identify the distinct academic/computer-science subjects actually present in
the source. Examples include Python, C, C++, DBMS, Data Structures & Algorithms,
Operating System, Computer Networks, Computer Architecture, Software Engineering,
Web Technology, Artificial Intelligence, Machine Learning, Computer Graphics,
Cyber Security, etc. These are examples only; do not invent topics not supported
by the document.

Then assign EXACTLY ONE topic to every question. The topic must be one of the
topics detected from the document.

Classification rules:
- Python syntax, libraries, data types, functions, OOP in Python -> Python
- C language syntax, pointers, memory, functions, structures -> C
- C++ syntax, STL, classes, inheritance, templates -> C++
- SQL, normalization, transactions, keys, relational model -> DBMS
- arrays, linked lists, stacks, queues, trees, graphs, sorting/searching -> Data Structures & Algorithms
- processes, threads, scheduling, deadlocks, memory management -> Operating System
- TCP/IP, routing, protocols, LAN/WAN, network models -> Computer Networks

Do not force these example labels if the source uses a more appropriate subject.
Use the most precise topic supported by the source.

Rules:
1. Every question must be answerable from the supplied document.
2. Do not copy source sentences directly.
3. Exactly four options A-D.
4. Exactly one correct answer.
5. Plausible distractors.
6. Avoid ambiguity and duplicates.
7. Explanations must be concise and educational.
8. Do not invent unsupported facts.
9. Return only the requested structured JSON.
{focus}
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=f"SOURCE DOCUMENT:\n\n{trim_document(document_text)}",
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=QUESTION_SCHEMA,
            max_output_tokens=14000
        )
    )
    raw = (response.text or "").strip()
    if not raw:
        raise ValueError("Gemini returned an empty response.")
    data = json.loads(raw)
    return validate_generated(data, count)

def oid(value: str):
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid study ID.")

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("new.html")

@app.get("/api/health")
def health():
    try:
        mongo.admin.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "unavailable"
    return {"status": "healthy", "model": MODEL, "database": db_status}

@app.get("/api/studies")
def list_studies():
    docs = studies.find(
        {},
        {"pdf_file_id": 0, "text": 0, "questions": 0}
    ).sort("created_at", DESCENDING)
    result = []
    for d in docs:
        result.append({
            "id": str(d["_id"]),
            "name": d["name"],
            "topics": d.get("topics", []),
            "question_count": len(d.get("questions", [])),
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else None
        })
    return {"success": True, "studies": result}

@app.get("/api/studies/{study_id}")
def get_study(study_id: str):
    d = studies.find_one({"_id": oid(study_id)})
    if not d:
        raise HTTPException(status_code=404, detail="Study not found.")
    return {
        "success": True,
        "study": {
            "id": str(d["_id"]),
            "name": d["name"],
            "topics": d.get("topics", []),
            "questions": d.get("questions", []),
            "question_count": len(d.get("questions", [])),
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else None
        }
    }

@app.get("/api/studies/{study_id}/pdf")
def download_pdf(study_id: str):
    d = studies.find_one({"_id": oid(study_id)}, {"pdf_file_id": 1, "name": 1})
    if not d or not d.get("pdf_file_id"):
        raise HTTPException(status_code=404, detail="PDF not found.")
    try:
        grid_file = fs.get(d["pdf_file_id"])
    except Exception:
        raise HTTPException(status_code=404, detail="Stored PDF not found.")
    return StreamingResponse(
        grid_file,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{d.get("name","study.pdf")}"'}
    )

@app.post("/api/studies")
async def create_study(
    file: UploadFile = File(...),
    text: str = Form(...),
    count: int = Form(50),
    topic: str = Form("")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="The PDF does not contain enough readable text.")
    count = min(max(int(count), 1), 100)

    try:
        topics, questions = generate_questions(text, count, topic)
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")

        now = datetime.now(timezone.utc)
        grid_id = fs.put(
            pdf_bytes,
            filename=file.filename,
            content_type="application/pdf",
            uploaded_at=now
        )

        doc = {
            "name": file.filename,
            "pdf_file_id": grid_id,
            "text": text,
            "topics": topics,
            "questions": questions,
            "created_at": now
        }
        result = studies.insert_one(doc)
        return {
            "success": True,
            "study_id": str(result.inserted_id),
            "name": file.filename,
            "topics": topics,
            "question_count": len(questions),
            "questions": questions
        }
    except HTTPException:
        raise
    except Exception as error:
        print("CREATE STUDY ERROR:", repr(error))
        raise HTTPException(status_code=500, detail=f"Could not create study: {error}")

@app.post("/api/generate")
def generate_compat(request: GenerateRequest):
    try:
        topics, questions = generate_questions(request.text, request.count, request.topic)
        return {"success": True, "count": len(questions), "topics": topics, "questions": questions}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {error}")

@app.post("/api/studies/{study_id}/attempts")
def save_attempt(study_id: str, request: SaveAttemptRequest):
    if request.study_id != study_id:
        raise HTTPException(status_code=400, detail="Study ID mismatch.")
    if not studies.find_one({"_id": oid(study_id)}, {"_id": 1}):
        raise HTTPException(status_code=404, detail="Study not found.")
    doc = {
        "study_id": oid(study_id),
        "correct": request.correct,
        "wrong": request.wrong,
        "skipped": request.skipped,
        "accuracy": request.accuracy,
        "answers": request.answers,
        "created_at": datetime.now(timezone.utc)
    }
    attempts.insert_one(doc)
    return {"success": True}

@app.get("/api/studies/{study_id}/attempts")
def get_attempts(study_id: str):
    oid_value = oid(study_id)
    rows = attempts.find({"study_id": oid_value}).sort("created_at", DESCENDING).limit(50)
    result = []
    for d in rows:
        result.append({
            "id": str(d["_id"]),
            "correct": d.get("correct", 0),
            "wrong": d.get("wrong", 0),
            "skipped": d.get("skipped", 0),
            "accuracy": d.get("accuracy", 0),
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else None
        })
    return {"success": True, "attempts": result}

@app.delete("/api/studies/{study_id}")
def delete_study(study_id: str):
    study_oid = oid(study_id)
    d = studies.find_one({"_id": study_oid})
    if not d:
        raise HTTPException(status_code=404, detail="Study not found.")
    if d.get("pdf_file_id"):
        try:
            fs.delete(d["pdf_file_id"])
        except Exception:
            pass
    attempts.delete_many({"study_id": study_oid})
    studies.delete_one({"_id": study_oid})
    return {"success": True}

@app.post("/api/doubt")
def answer_doubt(request: DoubtRequest):
    history_text = ""
    for item in request.history[-8:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in ["user", "assistant"] and content:
            history_text += f"\n{role.upper()}: {content}"

    prompt = f"""
You are a patient and knowledgeable university tutor.

Question:
{request.question}

Options:
A) {request.options.get("A", "")}
B) {request.options.get("B", "")}
C) {request.options.get("C", "")}
D) {request.options.get("D", "")}

Correct answer:
{request.correctAnswer} — {request.options.get(request.correctAnswer, "")}

Student's answer:
{request.studentAnswer or "(not recorded)"}

Official explanation:
{request.explanation or "(none provided)"}

Previous conversation:
{history_text}

Current student doubt:
{request.doubt}

Stay focused on this question. Use simple language, correct misconceptions,
and normally answer in 2-5 sentences.
"""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=700)
        )
        answer = (response.text or "").strip() or "I couldn't generate an answer."
        return {"success": True, "answer": answer}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Could not answer the doubt: {error}")

if __name__ == "__main__":

    uvicorn.run("api.index:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
