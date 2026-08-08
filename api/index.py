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
MONGODB_DB = os.getenv("MONGODB_DB", "rsmssb_mcq")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RSMSSB Computer Instructor MCQ Platform",
    version="5.0.0",
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
# GLOBAL CLIENTS
#
# IMPORTANT:
# Do NOT connect to MongoDB during module import.
# Vercel imports this file when starting a serverless
# function. Database connections should be created lazily.
# ============================================================

gemini_client = None
mongo_client = None
mongo_db = None
studies_collection = None
attempts_collection = None
grid_fs = None


# ============================================================
# CLIENT INITIALIZATION
# ============================================================

def get_gemini_client():

    global gemini_client

    if gemini_client is not None:
        return gemini_client

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing."
        )

    gemini_client = genai.Client(
        api_key=api_key
    )

    return gemini_client


def get_database():

    global mongo_client
    global mongo_db
    global studies_collection
    global attempts_collection
    global grid_fs

    if mongo_db is not None:
        return mongo_db

    mongodb_uri = os.getenv("MONGODB_URI")

    if not mongodb_uri:
        raise RuntimeError(
            "MONGODB_URI is missing."
        )

    database_name = os.getenv(
        "MONGODB_DB",
        "rsmssb_mcq"
    )

    mongo_client = MongoClient(
        mongodb_uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=15000,
        maxPoolSize=10,
        minPoolSize=0,
    )

    mongo_db = mongo_client[
        database_name
    ]

    studies_collection = mongo_db[
        "studies"
    ]

    attempts_collection = mongo_db[
        "attempts"
    ]

    grid_fs = gridfs.GridFS(
        mongo_db
    )

    return mongo_db


def get_studies():

    get_database()

    return studies_collection


def get_attempts():

    get_database()

    return attempts_collection


def get_gridfs():

    get_database()

    return grid_fs


# ============================================================
# MODELS
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
# GEMINI RESPONSE SCHEMA
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
):

    if len(text) <= max_chars:
        return text

    third = max_chars // 3

    middle = len(text) // 2

    return (
        text[:third]
        + "\n\n[... middle section omitted ...]\n\n"
        + text[
            middle - third // 2:
            middle + third // 2
        ]
        + "\n\n[... later section omitted ...]\n\n"
        + text[-third:]
    )


def normalize_topic(value: str):

    return " ".join(
        str(value or "")
        .strip()
        .split()
    )


def object_id(value: str):

    try:

        return ObjectId(value)

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid study ID."
        )


# ============================================================
# VALIDATE GEMINI OUTPUT
# ============================================================

def validate_generated(
    data: Any,
    requested_count: int
):

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            "Gemini did not return a valid object."
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

        topic = normalize_topic(
            topic
        )

        if (
            topic
            and topic.lower()
            not in {
                x.lower()
                for x in topics
            }
        ):

            topics.append(topic)

    topic_lookup = {
        x.lower(): x
        for x in topics
    }

    valid_questions = []

    seen = set()

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

        clean_options = {}

        for letter in [
            "A",
            "B",
            "C",
            "D"
        ]:

            value = str(
                options.get(
                    letter,
                    ""
                )
            ).strip()

            if not value:
                break

            clean_options[
                letter
            ] = value

        if len(clean_options) != 4:
            continue

        if not topic:
            continue

        if (
            topic.lower()
            not in topic_lookup
        ):

            topics.append(topic)

            topic_lookup[
                topic.lower()
            ] = topic

        canonical_topic = topic_lookup[
            topic.lower()
        ]

        key = question.casefold()

        if key in seen:
            continue

        seen.add(key)

        valid_questions.append({

            "question": question,

            "options": clean_options,

            "correctAnswer": correct,

            "explanation": explanation,

            "topic": canonical_topic

        })

        if len(valid_questions) >= requested_count:
            break

    if not valid_questions:

        raise ValueError(
            "Gemini did not generate any valid questions."
        )

    return (
        topics,
        valid_questions
    )


# ============================================================
# GEMINI CALL
# ============================================================

def generate_questions(
    document_text: str,
    count: int,
    focus_topic: str = ""
):

    gemini = get_gemini_client()

    focus_instruction = ""

    if focus_topic.strip():

        focus_instruction = f"""

FOCUS TOPIC:

{focus_topic.strip()}

Generate questions ONLY about this topic.

If the topic is not supported by the
document, do not invent information.

"""


    system_instruction = f"""

You are an expert question writer for
the RSMSSB Computer Instructor examination.

Create up to {count} original MCQs strictly
from the supplied study material.

==================================================
SUBJECT IDENTIFICATION
==================================================

First identify the distinct academic/computer
science subjects actually present in the source.

Possible subjects include:

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
etc.

These are examples only.

Do NOT invent a subject that is not supported
by the source document.

==================================================
QUESTION CLASSIFICATION
==================================================

Every question MUST have exactly ONE topic.

Examples:

Python syntax, Python data types,
functions, classes, OOP
→ Python

C syntax, pointers, structures,
memory and functions
→ C

C++ classes, inheritance, STL,
templates
→ C++

SQL, normalization, transactions,
keys, relational model
→ DBMS

Arrays, linked lists, stacks,
queues, trees, graphs,
sorting and searching
→ Data Structures & Algorithms

Processes, threads, scheduling,
deadlocks and memory management
→ Operating System

TCP/IP, routing, protocols,
LAN/WAN and network models
→ Computer Networks

Use a more precise topic when the
document supports one.

==================================================
MCQ RULES
==================================================

1. Every question must be answerable
   from the supplied document.

2. Do not copy source sentences.

3. Create original questions.

4. Exactly four options:
   A, B, C and D.

5. Exactly one correct answer.

6. Distractors must be plausible.

7. Avoid ambiguity.

8. Avoid duplicate questions.

9. Explanations must be concise.

10. Do not invent unsupported facts.

11. Every question must have exactly
    one topic.

12. Topic must be supported by the
    source document.

13. Return ONLY structured JSON.

{focus_instruction}
"""


    prompt = f"""

SOURCE DOCUMENT:

{trim_document(document_text)}

"""


    last_error = None

    for attempt in range(3):

        try:

            response = gemini.models.generate_content(

                model=MODEL,

                contents=prompt,

                config=types.GenerateContentConfig(

                    system_instruction=(
                        system_instruction
                    ),

                    response_mime_type=(
                        "application/json"
                    ),

                    response_schema=(
                        QUESTION_SCHEMA
                    ),

                    max_output_tokens=14000
                )
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

        except Exception as error:

            last_error = error

            message = str(
                error
            ).lower()

            temporary = (
                "503" in message
                or "unavailable" in message
                or "high demand" in message
                or "429" in message
                or "rate limit" in message
                or "timeout" in message
            )

            if not temporary:
                raise

            if attempt < 2:

                wait_seconds = (
                    2 ** attempt
                )

                print(
                    "Gemini temporary error. "
                    f"Retrying in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

    raise last_error


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def root():

    frontend = (
        Path(__file__)
        .resolve()
        .parent
        .parent
        / "new.html"
    )

    if not frontend.exists():

        return {
            "status": "ok",
            "message": (
                "RSMSSB MCQ API is running. "
                "new.html was not found."
            )
        }

    return FileResponse(
        str(frontend),
        media_type="text/html"
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

        "success": True,

        "message": (
            "RSMSSB Computer Instructor "
            "MCQ Platform API"
        ),

        "model": MODEL

    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/api/health"
)
def health():

    database = "not_checked"

    try:

        get_database()

        mongo_client.admin.command(
            "ping"
        )

        database = "connected"

    except Exception as error:

        print(
            "HEALTH MONGODB ERROR:",
            repr(error)
        )

        database = "unavailable"

    return {

        "status": "healthy",

        "model": MODEL,

        "database": database

    }


# ============================================================
# LIST STUDIES
# ============================================================

@app.get(
    "/api/studies"
)
def list_studies():

    try:

        collection = get_studies()

        documents = collection.find(
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

        studies_result = []

        for document in documents:

            created_at = (
                document.get(
                    "created_at"
                )
            )

            studies_result.append({

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
                    created_at.isoformat()
                    if created_at
                    else None
                )

            })

        return {

            "success": True,

            "studies": studies_result

        }

    except Exception as error:

        print(
            "LIST STUDIES ERROR:",
            repr(error)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Could not load studies: "
                f"{error}"
            )

        )


# ============================================================
# GET STUDY
# ============================================================

@app.get(
    "/api/studies/{study_id}"
)
def get_study(
    study_id: str
):

    collection = get_studies()

    document = collection.find_one({

        "_id": object_id(
            study_id
        )

    })

    if not document:

        raise HTTPException(
            status_code=404,
            detail="Study not found."
        )

    created_at = (
        document.get(
            "created_at"
        )
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
                created_at.isoformat()
                if created_at
                else None
            )

        }

    }


# ============================================================
# PDF
# ============================================================

@app.get(
    "/api/studies/{study_id}/pdf"
)
def download_pdf(
    study_id: str
):

    collection = get_studies()

    document = collection.find_one(

        {
            "_id": object_id(
                study_id
            )
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

    file_id = document.get(
        "pdf_file_id"
    )

    if not file_id:

        raise HTTPException(
            status_code=404,
            detail="PDF not found."
        )

    try:

        stored_file = get_gridfs().get(
            file_id
        )

    except Exception as error:

        print(
            "GRIDFS GET ERROR:",
            repr(error)
        )

        raise HTTPException(
            status_code=404,
            detail="Stored PDF not found."
        )

    filename = document.get(
        "name",
        "study.pdf"
    )

    return StreamingResponse(

        stored_file,

        media_type="application/pdf",

        headers={
            "Content-Disposition":
            f'inline; filename="{filename}"'
        }

    )


# ============================================================
# CREATE STUDY
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

        count = int(
            count
        )

    except Exception:

        count = 50

    count = min(
        max(count, 1),
        100
    )

    try:

        print(
            f"Generating {count} questions "
            f"from {filename}"
        )

        topics, questions = (
            generate_questions(

                document_text=text,

                count=count,

                focus_topic=topic

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

        # Get database only now
        collection = get_studies()

        # Store PDF
        file_id = get_gridfs().put(

            pdf_bytes,

            filename=filename,

            content_type=(
                "application/pdf"
            ),

            uploaded_at=now

        )

        document = {

            "name": filename,

            "pdf_file_id": file_id,

            "text": text,

            "topics": topics,

            "questions": questions,

            "created_at": now

        }

        result = collection.insert_one(
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
            "================================"
        )

        print(
            "CREATE STUDY ERROR:"
        )

        print(
            repr(error)
        )

        print(
            "================================"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Could not create study: "
                f"{error}"
            )

        )


# ============================================================
# GENERATE COMPATIBILITY ENDPOINT
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

                document_text=request.text,

                count=request.count,

                focus_topic=request.topic

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
# SAVE ATTEMPT
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

    study_oid = object_id(
        study_id
    )

    collection = get_studies()

    if not collection.find_one(

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

    get_attempts().insert_one(
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

    study_oid = object_id(
        study_id
    )

    documents = (
        get_attempts()
        .find(
            {
                "study_id": study_oid
            }
        )
        .sort(
            "created_at",
            DESCENDING
        )
        .limit(50)
    )

    result = []

    for document in documents:

        created_at = (
            document.get(
                "created_at"
            )
        )

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
                created_at.isoformat()
                if created_at
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

    study_oid = object_id(
        study_id
    )

    collection = get_studies()

    document = collection.find_one({

        "_id": study_oid

    })

    if not document:

        raise HTTPException(

            status_code=404,

            detail="Study not found."

        )

    file_id = document.get(
        "pdf_file_id"
    )

    if file_id:

        try:

            get_gridfs().delete(
                file_id
            )

        except Exception as error:

            print(
                "GRIDFS DELETE WARNING:",
                repr(error)
            )

    get_attempts().delete_many({

        "study_id": study_oid

    })

    collection.delete_one({

        "_id": study_oid

    })

    return {
        "success": True
    }


# ============================================================
# AI DOUBT
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

Question:

{request.question}

Options:

A) {request.options.get("A", "")}

B) {request.options.get("B", "")}

C) {request.options.get("C", "")}

D) {request.options.get("D", "")}

Correct answer:

{request.correctAnswer}
-
{request.options.get(
    request.correctAnswer,
    ""
)}

Student's answer:

{request.studentAnswer or "(not recorded)"}

Official explanation:

{request.explanation or "(none provided)"}

Previous conversation:

{history_text}

Current student doubt:

{request.doubt}

Stay focused on this question.

Use simple language.

Correct misconceptions.

Normally answer in 2-5 sentences.

"""

    try:

        gemini = get_gemini_client()

        response = gemini.models.generate_content(

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
            "DOUBT ERROR:",
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
# LOCAL DEVELOPMENT ONLY
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
