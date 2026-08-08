import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Create a .env file and add your Gemini API key."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# Current stable Gemini Flash model.
# Gemini 3.6 Flash supports structured JSON output.
MODEL = "gemini-3.6-flash"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Mock Paper Generator - Gemini",
    version="3.0.0"
)

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
    text: str = Field(..., min_length=20)
    count: int = Field(default=15, ge=1, le=30)
    topic: str = ""


class DoubtRequest(BaseModel):
    question: str
    options: Dict[str, str]
    correctAnswer: str
    studentAnswer: str = ""
    explanation: str = ""
    doubt: str
    history: List[Dict[str, str]] = []


# ============================================================
# STRUCTURED MCQ SCHEMA
# ============================================================

QUESTION_SCHEMA = {
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
            "explanation": {
                "type": "STRING"
            }
        },
        "required": [
            "question",
            "options",
            "correctAnswer",
            "explanation"
        ]
    }
}


# ============================================================
# HELPERS
# ============================================================

def validate_questions(data: Any) -> List[Dict[str, Any]]:
    """Validate Gemini's generated MCQs."""

    if not isinstance(data, list):
        raise ValueError("Gemini did not return a question array.")

    valid = []
    seen = set()

    for item in data:

        if not isinstance(item, dict):
            continue

        question = str(
            item.get("question", "")
        ).strip()

        options = item.get("options", {})
        correct = item.get("correctAnswer", "")
        explanation = str(
            item.get("explanation", "")
        ).strip()

        if not question:
            continue

        if not isinstance(options, dict):
            continue

        if correct not in ["A", "B", "C", "D"]:
            continue

        normalized = {
            key: str(
                options.get(key, "")
            ).strip()
            for key in ["A", "B", "C", "D"]
        }

        if any(
            not normalized[key]
            for key in ["A", "B", "C", "D"]
        ):
            continue

        duplicate_key = question.lower()

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)

        valid.append({
            "question": question,
            "options": normalized,
            "correctAnswer": correct,
            "explanation": explanation
        })

    return valid


def trim_document(text: str, max_chars: int = 90000) -> str:
    """
    Keep the document within a practical request size while preserving
    beginning, middle and end.
    """

    if len(text) <= max_chars:
        return text

    third = max_chars // 3
    middle = len(text) // 2

    return (
        text[:third]
        + "\n\n[...middle section omitted for context size...]\n\n"
        + text[
            middle - third // 2:
            middle + third // 2
        ]
        + "\n\n[...later section...]\n\n"
        + text[-third:]
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Mock Paper Generator API using Gemini 3.6 Flash is running."
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("new.html")


# ============================================================
# GENERATE QUESTIONS
# ============================================================

@app.post("/api/generate")
def generate_questions(request: GenerateRequest):

    document_text = request.text.strip()

    if len(document_text) < 20:
        raise HTTPException(
            status_code=400,
            detail="The PDF does not contain enough readable text."
        )

    count = min(max(request.count, 1), 30)

    topic_instruction = ""

    if request.topic.strip():

        topic_instruction = f"""
FOCUS TOPIC:

{request.topic.strip()}

Generate questions only about this topic.
If this topic is not present in the source material,
return an empty array.
"""

    system_instruction = f"""
You are an expert university examination question writer.

Create a fresh mock examination from the supplied study material.

Generate up to {count} high-quality multiple-choice questions.

{topic_instruction}

RULES:

1. Every question must be answerable from the supplied document.
2. Do not copy sentences directly from the source.
3. Write original question wording.
4. Test understanding, concepts, definitions, comparisons,
   processes, applications and important facts.
5. Every question must have exactly four options.
6. Options must be A, B, C and D.
7. Exactly one option must be correct.
8. Incorrect options should be plausible.
9. Avoid ambiguous questions.
10. Do not repeat or closely rephrase questions.
11. Keep explanations concise and educational.
12. Do not invent facts that are not supported by the source.
13. Return only the structured JSON array.
"""

    document_text = trim_document(document_text)

    prompt = f"""
SOURCE DOCUMENT:

{document_text}
"""

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=QUESTION_SCHEMA,
                max_output_tokens=12000
            )
        )

        raw = (response.text or "").strip()

        if not raw:
            raise ValueError(
                "Gemini returned an empty response."
            )

        parsed = json.loads(raw)

        questions = validate_questions(parsed)

        if not questions:
            raise ValueError(
                "Gemini did not generate any valid questions."
            )

        return {
            "success": True,
            "count": len(questions),
            "questions": questions
        }

    except Exception as error:

        print("\n========== GEMINI GENERATION ERROR ==========")
        print(repr(error))
        print("=============================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed: {str(error)}"
        )


# ============================================================
# AI DOUBT TUTOR
# ============================================================

@app.post("/api/doubt")
def answer_doubt(request: DoubtRequest):

    options = request.options

    history_text = ""

    for item in request.history[-8:]:

        role = item.get("role", "")
        content = item.get("content", "")

        if role in ["user", "assistant"] and content:
            history_text += (
                f"\n{role.upper()}: {content}"
            )

    prompt = f"""
You are a patient and knowledgeable university tutor.

The student is reviewing this MCQ:

Question:
{request.question}

Options:
A) {options.get("A", "")}
B) {options.get("B", "")}
C) {options.get("C", "")}
D) {options.get("D", "")}

Correct answer:
{request.correctAnswer} — {options.get(request.correctAnswer, "")}

Student's answer:
{request.studentAnswer or "(not recorded)"}

Official explanation:
{request.explanation or "(none provided)"}

Previous conversation:
{history_text}

Current student doubt:
{request.doubt}

Rules:
1. Stay focused on this question and its underlying concept.
2. Use simple language.
3. Correct misconceptions if necessary.
4. Refer to the options when useful.
5. Normally answer in 2-5 sentences.
6. Do not contradict the supplied question context.
"""

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=700
            )
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = (
                "I couldn't generate an explanation. "
                "Please try rephrasing your doubt."
            )

        return {
            "success": True,
            "answer": answer
        }

    except Exception as error:

        print("\n========== GEMINI DOUBT ERROR ==========")
        print(repr(error))
        print("========================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Could not answer the doubt: {str(error)}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    import os

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )
