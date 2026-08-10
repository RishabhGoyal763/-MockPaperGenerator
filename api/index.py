import os
import json
import random
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import certifi
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
import gridfs
import uvicorn

# NOTE: this API no longer accepts multipart/form-data uploads at all —
# the PDF and its extracted text go straight from the browser to Vercel
# Blob storage (see api/blob-upload.js + index.html), and /api/studies only
# receives a small JSON body with the resulting URLs. That sidesteps two
# stacked limits that used to break large uploads here:
#   1. Starlette's MultiPartParser caps each multipart part at 1MB by
#      default since 0.40, and setting MultiPartParser.max_part_size as a
#      class attribute (the old workaround) stopped being honored in
#      Starlette 0.44+ (starlette #2815) — patching it here no longer helps.
#   2. Vercel Functions hard-cap every request body at 4.5MB regardless of
#      language/runtime and regardless of any app-level config, so even a
#      correctly patched Starlette would never get you past that.
# If you ever add another endpoint that must accept multipart/form-data
# directly, pin `starlette<0.44` in requirements.txt or use
# `await request.form(max_part_size=...)` inside the route itself —
# the class-attribute trick is no longer reliable.

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

mongo = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000,
    tlsCAFile=certifi.where()
)
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
    exclude: List[str] = []

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

def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))

def split_into_chunks(text: str, target_chunk_bytes: int = 6000) -> List[str]:
    """Break document text into paragraph-aligned chunks so we can randomly
    sample sections of it instead of always using the same fixed slice.
    Sized by UTF-8 byte length, since non-ASCII text (e.g. Hindi/Devanagari)
    can be 2-3 bytes per character."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""
    current_bytes = 0
    for p in paragraphs:
        p_bytes = utf8_len(p)
        if current and current_bytes + p_bytes + 2 > target_chunk_bytes:
            chunks.append(current)
            current = p
            current_bytes = p_bytes
        else:
            current = f"{current}\n\n{p}" if current else p
            current_bytes += p_bytes + (2 if current_bytes else 0)
    if current:
        chunks.append(current)
    return chunks

def sample_document_text(text: str, max_bytes: int = 60000) -> str:
    """Return a version of the document under max_bytes (UTF-8 encoded).
    If the document is short enough, return it whole. Otherwise randomly
    sample chunks (kept in original order) each call, so repeated
    generations draw from different parts of the source material instead
    of always the same fixed portion. Kept well below Gemini's ~1MB
    per-request-Part limit, with margin for non-ASCII text."""
    if utf8_len(text) <= max_bytes:
        return text

    chunks = split_into_chunks(text)
    if len(chunks) <= 1:
        # Fall back to a random contiguous character window; character
        # count is a safe (over-)estimate of a byte budget here.
        start = random.randint(0, max(0, len(text) - max_bytes))
        window = text[start:start + max_bytes]
        while utf8_len(window) > max_bytes and window:
            window = window[:-1]
        return window

    order = list(range(len(chunks)))
    random.shuffle(order)

    selected = []
    total = 0
    for idx in order:
        chunk_bytes = utf8_len(chunks[idx])
        if selected and total + chunk_bytes > max_bytes:
            continue
        selected.append(idx)
        total += chunk_bytes
        if total >= max_bytes:
            break

    selected.sort()
    pieces = []
    prev_idx = None
    for idx in selected:
        if prev_idx is not None and idx != prev_idx + 1:
            pieces.append("[...section omitted...]")
        pieces.append(chunks[idx])
        prev_idx = idx
    return "\n\n".join(pieces)

def normalize_topic(value: str) -> str:
    return " ".join(str(value or "").strip().split())

def validate_generated(data: Any, requested_count: int, exclude_keys: Optional[set] = None):
    exclude_keys = exclude_keys or set()
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
        if key in seen or key in exclude_keys:
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

def generate_questions(document_text: str, count: int, focus_topic: str = "", exclude_questions: Optional[List[str]] = None):
    exclude_questions = [str(q).strip() for q in (exclude_questions or []) if str(q).strip()]
    exclude_keys = {q.casefold() for q in exclude_questions}

    focus = ""
    if focus_topic.strip():
        focus = f"""
FOCUS TOPIC:
{focus_topic.strip()}
Generate questions only about this topic. If it is not supported by the source,
return no questions.
"""

    exclude_block = ""
    if exclude_questions:
        # Cap the list sent to the model, both by count and by byte size, so
        # this system_instruction Part stays well under Gemini's request
        # limits even after many "generate another set" clicks.
        listed = []
        running_bytes = 0
        max_exclude_bytes = 12000
        for q in reversed(exclude_questions[-100:]):
            q_bytes = utf8_len(q) + 3
            if listed and running_bytes + q_bytes > max_exclude_bytes:
                break
            listed.append(q)
            running_bytes += q_bytes
        listed.reverse()
        joined = "\n".join(f"- {q}" for q in listed)
        exclude_block = f"""
DO NOT REPEAT ANY OF THESE PREVIOUSLY GENERATED QUESTIONS.
Do not reuse their wording, and do not simply reword the same fact being tested.
Cover different facts, sections, or angles of the source material instead.
Previously generated questions:
{joined}
"""

    session_nonce = uuid.uuid4().hex[:8]

    system = f"""
SESSION: {session_nonce}
This is a fresh generation request. Vary your selection of facts, question
ordering, and phrasing from any previous run on this same source document.
{exclude_block}
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
    # Scale the output budget with how many questions were requested, so a
    # large count against a large source document doesn't get its JSON
    # response cut off mid-way.
    output_tokens = min(32000, 3000 + count * 220)

    # Start well under the ~1MB request-size ceiling (leaving headroom for
    # the system prompt, schema, and exclude list riding in the same
    # request), and automatically shrink and retry if the API still
    # rejects it for size, rather than surfacing that error to the user.
    budget_attempts = [220_000, 120_000, 60_000, 25_000]
    last_error = None

    for attempt_bytes in budget_attempts:
        source_text = sample_document_text(document_text, max_bytes=attempt_bytes)
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=f"SOURCE DOCUMENT:\n\n{source_text}",
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=QUESTION_SCHEMA,
                    max_output_tokens=output_tokens,
                    temperature=1.2,
                    top_p=0.97,
                    top_k=64
                )
            )
            raw = (response.text or "").strip()
            if not raw:
                raise ValueError("Gemini returned an empty response.")
            data = json.loads(raw)
            return validate_generated(data, count, exclude_keys)
        except Exception as error:
            message = str(error).lower()
            is_size_error = "exceeded maximum size" in message or "1024kb" in message or "1048576" in message
            last_error = error
            if not is_size_error:
                raise
            # else: too big even at this budget, try the next smaller one

    raise last_error

def oid(value: str):
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid study ID.")

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

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

class CreateStudyFromBlob(BaseModel):
    pdf_url: str
    text_url: str
    filename: str
    count: int = 50
    topic: str = ""

@app.post("/api/studies")
async def create_study(payload: CreateStudyFromBlob):
    # The browser has already uploaded the PDF and its extracted text
    # straight to Vercel Blob (see /api/blob-upload + index.html). This
    # endpoint only ever receives a small JSON body with the resulting
    # URLs, so it's never subject to Vercel's 4.5MB function body limit or
    # Starlette's multipart part-size limit — both of which only apply to
    # data sent IN the request body.
    if not payload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    count = min(max(int(payload.count), 1), 100)

    try:
        async with httpx.AsyncClient(timeout=120.0) as http_client:
            pdf_response = await http_client.get(payload.pdf_url)
            pdf_response.raise_for_status()
            pdf_bytes = pdf_response.content

            text_response = await http_client.get(payload.text_url)
            text_response.raise_for_status()
            text = text_response.text

        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")
        if len(text.strip()) < 20:
            raise HTTPException(status_code=400, detail="The PDF does not contain enough readable text.")

        topics, questions = generate_questions(text, count, payload.topic)

        now = datetime.now(timezone.utc)
        grid_id = fs.put(
            pdf_bytes,
            filename=payload.filename,
            content_type="application/pdf",
            uploaded_at=now
        )

        doc = {
            "name": payload.filename,
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
            "name": payload.filename,
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
        topics, questions = generate_questions(
            request.text, request.count, request.topic, request.exclude
        )
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
