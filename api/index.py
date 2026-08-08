import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
)

from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from pymongo import MongoClient, DESCENDING
from bson import ObjectId
import gridfs


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MONGODB_URI = os.getenv("MONGODB_URI")

MONGODB_DB = os.getenv(
    "MONGODB_DB",
    "rsmssb_mcq"
)

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ============================================================
# REQUIRED ENVIRONMENT VARIABLES
# ============================================================

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Add it to Vercel Environment Variables."
    )

if not MONGODB_URI:
    raise RuntimeError(
        "MONGODB_URI is missing. "
        "Add it to Vercel Environment Variables."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# MONGODB
# ============================================================

mongo = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
)

db = mongo[MONGODB_DB]

studies = db["studies"]

attempts = db["attempts"]

fs = gridfs.GridFS(db)


# Create index
try:
    studies.create_index(
        [("created_at", DESCENDING)]
    )
except Exception as error:
    print(
        "MongoDB index creation warning:",
        repr(error)
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="RSMSSB Computer Instructor MCQ Platform",
    version="4.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class GenerateRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=20
    )

    count: int = Field(
        default=50,
        ge=1,
        le=100
    )

    topic: str = ""


class DoubtRequest(BaseModel):

    question: str

    options: Dict[str, str]

    correctAnswer: str

    studentAnswer: str = ""

    explanation: str = ""

    doubt: str

    history: List[Dict[str, str]] = Field(
        default_factory=list
    )


class SaveAttemptRequest(BaseModel):

    study_id: str

    correct: int

    wrong: int

    skipped: int

    accuracy: int

    answers: List[Dict[str, Any]] = Field(
        default_factory=list
    )


# ============================================================
# GEMINI STRUCTURED OUTPUT SCHEMA
# ============================================================

QUESTION_SCHEMA = {

    "type": "OBJECT",

    "properties": {

        "topics": {

            "type": "ARRAY",

            "items": {
                "type": "STRING"
            }

        },

        "questions": {

            "type": "ARRAY",

            "items": {

                "type": "OBJECT",

                "properties": {

                    "question": {
                        "type": "STRING"
                    },

                    "options": {

                        "type": "OBJECT",

                        "properties": {

                            "A": {
                                "type": "STRING"
                            },

                            "B": {
                                "type": "STRING"
                            },

                            "C": {
                                "type": "STRING"
                            },

                            "D": {
                                "type": "STRING"
                            }

                        },

                        "required": [
                            "A",
                            "B",
                            "C",
                            "D"
                        ]

                    },

                    "correctAnswer": {

                        "type": "STRING",

                        "enum": [
                            "A",
                            "B",
                            "C",
                            "D"
                        ]

                    },

                    "explanation": {
                        "type": "STRING"
                    },

                    "topic": {
                        "type": "STRING"
                    }

                },

                "required": [
                    "question",
                    "options",
                    "correctAnswer",
                    "explanation",
                    "topic"
                ]

            }

        }

    },

    "required": [
        "topics",
        "questions"
    ]

}


# ============================================================
# HELPERS
# ============================================================

def trim_document(
    text: str,
    max_chars: int = 90000
) -> str:

    if len(text) <= max_chars:
        return text

    third = max_chars // 3

    middle = len(text) // 2

    return (
        text[:third]
        + "\n\n"
        "[... middle section omitted ...]"
        "\n\n"
        + text[
            middle - third // 2:
            middle + third // 2
        ]
        + "\n\n"
        "[... later section ...]"
        "\n\n"
        + text[-third:]
    )


def normalize_topic(value: str) -> str:

    return " ".join(
        str(value or "")
        .strip()
        .split()
    )


def validate_generated(
    data: Any,
    requested_count: int
):

    if not isinstance(data, dict):

        raise ValueError(
            "Gemini did not return the expected object."
        )

    raw_topics = data.get(
        "topics",
        []
    )

    raw_questions = data.get(
        "questions",
        []
    )

    topics = []

    for topic in raw_topics:

        topic = normalize_topic(topic)

        if (
            topic
            and topic.lower()
            not in {
                x.lower()
                for x in topics
            }
        ):

            topics.append(topic)

    valid = []

    seen = set()

    topic_lookup = {
        topic.lower(): topic
        for topic in topics
    }

    for item in raw_questions:

        if not isinstance(
            item,
            dict
        ):
            continue

        question = str(
            item.get(
                "question",
                ""
            )
        ).strip()

        options = item.get(
            "options",
            {}
        )

        correct = item.get(
            "correctAnswer",
            ""
        )

        explanation = str(
            item.get(
                "explanation",
                ""
            )
        ).strip()

        topic = normalize_topic(
            item.get(
                "topic",
                ""
            )
        )

        if not question:
            continue

        if not isinstance(
            options,
            dict
        ):
            continue

        if correct not in [
            "A",
            "B",
            "C",
            "D"
        ]:
            continue

        opts = {
            key: str(
                options.get(
                    key,
                    ""
                )
            ).strip()
            for key in [
                "A",
                "B",
                "C",
                "D"
            ]
        }

        if any(
            not opts[key]
            for key in opts
        ):
            continue

        if not topic:
            continue

        canonical = topic_lookup.get(
            topic.lower()
        )

        if canonical is None:

            topics.append(topic)

            topic_lookup[
                topic.lower()
            ] = topic

            canonical = topic

        duplicate_key = question.casefold()

        if duplicate_key in seen:
            continue

        seen.add(
            duplicate_key
        )

        valid.append({

            "question": question,

            "options": opts,

            "correctAnswer": correct,

            "explanation": explanation,

            "topic": canonical

        })

        if len(valid) >= requested_count:
            break

    if not valid:

        raise ValueError(
            "Gemini did not generate any valid questions."
        )

    return topics, valid


# ============================================================
# GEMINI GENERATION
# ============================================================

def call_gemini(
    prompt: str,
    system_instruction: str,
    max_output_tokens: int = 14000,
    retries: int = 3
):

    last_error = None

    for attempt in range(retries):

        try:

            response = client.models.generate_content(

                model=MODEL,

                contents=prompt,

                config=types.GenerateContentConfig(

                    system_instruction=system_instruction,

                    response_mime_type="application/json",

                    response_schema=QUESTION_SCHEMA,

                    max_output_tokens=max_output_tokens

                )

            )

            return response

        except Exception as error:

            last_error = error

            error_text = str(error).lower()

            temporary_error = (
                "503" in error_text
                or "unavailable" in error_text
                or "high demand" in error_text
                or "429" in error_text
                or "rate limit" in error_text
            )

            if not temporary_error:

                raise

            if attempt < retries - 1:

                wait_time = 2 ** attempt

                print(
                    f"Gemini temporary error. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

    raise last_error


# ============================================================
# GENERATE QUESTIONS
# ============================================================

def generate_questions(
    document_text: str,
    count: int,
    focus_topic: str = ""
):

    focus = ""

    if focus_topic.strip():

        focus = f"""

FOCUS TOPIC:

{focus_topic.strip()}

Generate questions ONLY about this topic.

If this topic is not supported by the
source document, return no questions.

"""


    system = f"""

You are an expert question writer for
the RSMSSB Computer Instructor examination.

Create up to {count} original MCQs strictly
from the supplied study material.

============================================================
STEP 1 — IDENTIFY SUBJECTS
============================================================

First identify the distinct academic/computer
science subjects actually present in the source.

Examples include:

Python
C
C++
DBMS
Data Structures & Algorithms
Operating System
Computer Networks
Computer Architecture
Software Engineering
Web Technology
Artificial Intelligence
Machine Learning
Computer Graphics
Cyber Security
Digital Logic
Theory of Computation
Information Technology
Rajasthan Computer Knowledge
etc.

These are examples only.

DO NOT invent topics that are not supported
by the source document.

============================================================
STEP 2 — CLASSIFY EVERY QUESTION
============================================================

Every question MUST have exactly ONE topic.

Examples:

Python syntax, Python data types,
Python functions, Python OOP
→ Python

C syntax, pointers, memory,
structures, functions
→ C

C++ classes, inheritance,
templates, STL
→ C++

SQL, normalization, transactions,
keys, relational model
→ DBMS

Arrays, linked lists, stacks,
queues, trees, graphs,
sorting and searching
→ Data Structures & Algorithms

Processes, threads, scheduling,
deadlocks, memory management
→ Operating System

TCP/IP, routing, protocols,
LAN/WAN, network models
→ Computer Networks

Do not force these labels if another
more precise subject is supported by
the document.

============================================================
QUESTION RULES
============================================================

1. Every question must be answerable
   from the supplied document.

2. Do not copy sentences directly
   from the source.

3. Write original question wording.

4. Test understanding, concepts,
   definitions, comparisons,
   processes, applications and
   important facts.

5. Every question must have exactly
   four options.

6. Options must be A, B, C and D.

7. Exactly one option must be correct.

8. Incorrect options must be plausible.

9. Avoid ambiguous questions.

10. Do not repeat questions.

11. Keep explanations concise
    and educational.

12. Do not invent unsupported facts.

13. Assign exactly one topic to
    every question.

14. The topic must be supported
    by the supplied document.

15. Return ONLY the structured JSON.

{focus}

"""


    prompt = f"""

SOURCE DOCUMENT:

{trim_document(document_text)}

"""


    response = call_gemini(

        prompt=prompt,

        system_instruction=system,

        max_output_tokens=14000,

        retries=3

    )


    raw = (
        response.text or ""
    ).strip()


    if not raw:

        raise ValueError(
            "Gemini returned an empty response."
        )


    data = json.loads(
        raw
    )


    return validate_generated(
        data,
        count
    )


# ============================================================
# OBJECT ID HELPER
# ============================================================

def oid(value: str):

    try:

        return ObjectId(value)

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid study ID."
        )


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def home():

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    frontend_path = (
        project_root
        / "new.html"
    )

    if not frontend_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "new.html not found "
                "in the project root."
            )
        )

    return FileResponse(
        str(frontend_path)
    )


# ============================================================
# API ROOT
# ============================================================

@app.get(
    "/api",
    include_in_schema=False
)
def api_root():

    return {

        "status": "ok",

        "message": (
            "RSMSSB Computer Instructor "
            f"MCQ Platform using {MODEL}"
        )

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/health"
)
def health():

    database_status = "unknown"

    try:

        mongo.admin.command(
            "ping"
        )

        database_status = "connected"

    except Exception:

        database_status = "unavailable"


    return {

        "status": "healthy",

        "model": MODEL,

        "database": database_status

    }


# ============================================================
# LIST STUDIES
# ============================================================

@app.get(
    "/api/studies"
)
def list_studies():

    try:

        docs = studies.find(
            {},
            {
                "pdf_file_id": 0,
                "text": 0,
                "questions": 0
            }
        ).sort(
            "created_at",
            DESCENDING
        )

        result = []

        for document in docs:

            result.append({

                "id": str(
                    document["_id"]
                ),

                "name": document.get(
                    "name",
                    "Untitled Study"
                ),

                "topics": document.get(
                    "topics",
                    []
                ),

                "question_count": len(
                    document.get(
                        "questions",
                        []
                    )
                ),

                "created_at": (
                    document
                    .get("created_at")
                    .isoformat()
                    if document.get(
                        "created_at"
                    )
                    else None
                )

            })

        return {

            "success": True,

            "studies": result

        }

    except Exception as error:

        print(
            "LIST STUDIES ERROR:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not load studies: "
                f"{error}"
            )
        )


# ============================================================
# GET SINGLE STUDY
# ============================================================

@app.get(
    "/api/studies/{study_id}"
)
def get_study(
    study_id: str
):

    document = studies.find_one(
        {
            "_id": oid(study_id)
        }
    )

    if not document:

        raise HTTPException(
            status_code=404,
            detail="Study not found."
        )

    return {

        "success": True,

        "study": {

            "id": str(
                document["_id"]
            ),

            "name": document.get(
                "name",
                "Untitled Study"
            ),

            "topics": document.get(
                "topics",
                []
            ),

            "questions": document.get(
                "questions",
                []
            ),

            "question_count": len(
                document.get(
                    "questions",
                    []
                )
            ),

            "created_at": (
                document
                .get("created_at")
                .isoformat()
                if document.get(
                    "created_at"
                )
                else None
            )

        }

    }


# ============================================================
# DOWNLOAD ORIGINAL PDF
# ============================================================

@app.get(
    "/api/studies/{study_id}/pdf"
)
def download_pdf(
    study_id: str
):

    document = studies.find_one(
        {
            "_id": oid(study_id)
        },
        {
            "pdf_file_id": 1,
            "name": 1
        }
    )

    if not document:

        raise HTTPException(
            status_code=404,
            detail="Study not found."
        )

    if not document.get(
        "pdf_file_id"
    ):

        raise HTTPException(
            status_code=404,
            detail="PDF not found."
        )

    try:

        grid_file = fs.get(
            document[
                "pdf_file_id"
            ]
        )

    except Exception:

        raise HTTPException(
            status_code=404,
            detail="Stored PDF not found."
        )

    filename = document.get(
        "name",
        "study.pdf"
    )

    return StreamingResponse(

        grid_file,

        media_type="application/pdf",

        headers={
            "Content-Disposition":
            f'inline; filename="{filename}"'
        }

    )


# ============================================================
# CREATE PERMANENT STUDY
# ============================================================

@app.post(
    "/api/studies"
)
async def create_study(

    file: UploadFile = File(...),

    text: str = Form(...),

    count: int = Form(50),

    topic: str = Form("")

):

    filename = (
        file.filename or ""
    )

    if not filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files "
                "are supported."
            )
        )


    if len(
        text.strip()
    ) < 20:

        raise HTTPException(
            status_code=400,
            detail=(
                "The PDF does not contain "
                "enough readable text."
            )
        )


    try:

        count = int(count)

    except Exception:

        count = 50


    count = min(
        max(count, 1),
        100
    )


    try:

        print(
            "Generating questions..."
        )

        topics, questions = (
            generate_questions(
                text,
                count,
                topic
            )
        )


        print(
            f"Generated {len(questions)} questions."
        )


        pdf_bytes = await file.read()


        if not pdf_bytes:

            raise ValueError(
                "Uploaded PDF is empty."
            )


        now = datetime.now(
            timezone.utc
        )


        # Store PDF permanently
        grid_id = fs.put(

            pdf_bytes,

            filename=filename,

            content_type=(
                "application/pdf"
            ),

            uploaded_at=now

        )


        document = {

            "name": filename,

            "pdf_file_id": grid_id,

            "text": text,

            "topics": topics,

            "questions": questions,

            "created_at": now

        }


        result = studies.insert_one(
            document
        )


        return {

            "success": True,

            "study_id": str(
                result.inserted_id
            ),

            "name": filename,

            "topics": topics,

            "question_count": len(
                questions
            ),

            "questions": questions

        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            "\n========== CREATE STUDY ERROR =========="
        )

        print(
            repr(error)
        )

        print(
            "=========================================\n"
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Could not create study: "
                f"{error}"
            )

        )


# ============================================================
# COMPATIBILITY GENERATE ENDPOINT
# ============================================================

@app.post(
    "/api/generate"
)
def generate_compat(
    request: GenerateRequest
):

    try:

        topics, questions = (
            generate_questions(
                request.text,
                request.count,
                request.topic
            )
        )

        return {

            "success": True,

            "count": len(
                questions
            ),

            "topics": topics,

            "questions": questions

        }

    except Exception as error:

        print(
            "GENERATE ERROR:",
            repr(error)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Question generation failed: "
                f"{error}"
            )

        )


# ============================================================
# SAVE EXAM ATTEMPT
# ============================================================

@app.post(
    "/api/studies/{study_id}/attempts"
)
def save_attempt(

    study_id: str,

    request: SaveAttemptRequest

):

    if request.study_id != study_id:

        raise HTTPException(
            status_code=400,
            detail="Study ID mismatch."
        )


    study_oid = oid(
        study_id
    )


    if not studies.find_one(
        {
            "_id": study_oid
        },
        {
            "_id": 1
        }
    ):

        raise HTTPException(
            status_code=404,
            detail="Study not found."
        )


    document = {

        "study_id": study_oid,

        "correct": request.correct,

        "wrong": request.wrong,

        "skipped": request.skipped,

        "accuracy": request.accuracy,

        "answers": request.answers,

        "created_at": datetime.now(
            timezone.utc
        )

    }


    attempts.insert_one(
        document
    )


    return {
        "success": True
    }


# ============================================================
# GET ATTEMPTS
# ============================================================

@app.get(
    "/api/studies/{study_id}/attempts"
)
def get_attempts(
    study_id: str
):

    study_oid = oid(
        study_id
    )


    rows = attempts.find(
        {
            "study_id": study_oid
        }
    ).sort(
        "created_at",
        DESCENDING
    ).limit(50)


    result = []


    for document in rows:

        result.append({

            "id": str(
                document["_id"]
            ),

            "correct": document.get(
                "correct",
                0
            ),

            "wrong": document.get(
                "wrong",
                0
            ),

            "skipped": document.get(
                "skipped",
                0
            ),

            "accuracy": document.get(
                "accuracy",
                0
            ),

            "created_at": (
                document
                .get("created_at")
                .isoformat()
                if document.get(
                    "created_at"
                )
                else None
            )

        })


    return {

        "success": True,

        "attempts": result

    }


# ============================================================
# DELETE STUDY
# ============================================================

@app.delete(
    "/api/studies/{study_id}"
)
def delete_study(
    study_id: str
):

    study_oid = oid(
        study_id
    )


    document = studies.find_one(
        {
            "_id": study_oid
        }
    )


    if not document:

        raise HTTPException(
            status_code=404,
            detail="Study not found."
        )


    # Delete PDF from GridFS
    if document.get(
        "pdf_file_id"
    ):

        try:

            fs.delete(
                document[
                    "pdf_file_id"
                ]
            )

        except Exception as error:

            print(
                "GridFS delete warning:",
                repr(error)
            )


    # Delete attempts
    attempts.delete_many(
        {
            "study_id": study_oid
        }
    )


    # Delete study
    studies.delete_one(
        {
            "_id": study_oid
        }
    )


    return {
        "success": True
    }


# ============================================================
# AI DOUBT TUTOR
# ============================================================

@app.post(
    "/api/doubt"
)
def answer_doubt(
    request: DoubtRequest
):

    history_text = ""


    for item in request.history[-8:]:

        role = item.get(
            "role",
            ""
        )

        content = item.get(
            "content",
            ""
        )


        if (
            role in [
                "user",
                "assistant"
            ]
            and content
        ):

            history_text += (
                f"\n{role.upper()}: "
                f"{content}"
            )


    prompt = f"""

You are a patient and knowledgeable
university tutor.

The student is reviewing this MCQ.

QUESTION:

{request.question}

OPTIONS:

A) {request.options.get("A", "")}

B) {request.options.get("B", "")}

C) {request.options.get("C", "")}

D) {request.options.get("D", "")}

CORRECT ANSWER:

{request.correctAnswer}
—
{request.options.get(
    request.correctAnswer,
    ""
)}

STUDENT'S ANSWER:

{request.studentAnswer or "(not recorded)"}

OFFICIAL EXPLANATION:

{request.explanation or "(none provided)"}

PREVIOUS CONVERSATION:

{history_text}

CURRENT STUDENT DOUBT:

{request.doubt}

RULES:

1. Stay focused on this question.

2. Explain the underlying concept.

3. Use simple language.

4. Correct misconceptions.

5. Refer to the options when useful.

6. Normally answer in 2-5 sentences.

7. Do not contradict the supplied
   question context.

"""


    try:

        response = client.models.generate_content(

            model=MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                max_output_tokens=700

            )

        )


        answer = (
            response.text or ""
        ).strip()


        if not answer:

            answer = (
                "I couldn't generate "
                "an answer."
            )


        return {

            "success": True,

            "answer": answer

        }


    except Exception as error:

        print(
            "GEMINI DOUBT ERROR:",
            repr(error)
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Could not answer the doubt: "
                f"{error}"
            )

        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api.index:app",

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        ),

        reload=True

    )
