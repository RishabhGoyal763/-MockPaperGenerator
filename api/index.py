import os
import json
from pathlib import Path
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
        "Add GEMINI_API_KEY in Vercel Environment Variables."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# Gemini model
MODEL = "gemini-3.6-flash"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Mock Paper Generator - Gemini",
    version="4.0.0"
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

    # DEFAULT = 40
    # MAXIMUM = 100
    count: int = Field(
        default=40,
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
# VALIDATE QUESTIONS
# ============================================================

def validate_questions(
    data: Any
) -> List[Dict[str, Any]]:

    if not isinstance(data, list):

        raise ValueError(
            "Gemini did not return a question array."
        )

    valid = []

    seen = set()

    for item in data:

        if not isinstance(item, dict):
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

        # ----------------------------------------------------
        # Question validation
        # ----------------------------------------------------

        if not question:
            continue

        # ----------------------------------------------------
        # Options validation
        # ----------------------------------------------------

        if not isinstance(
            options,
            dict
        ):
            continue

        # ----------------------------------------------------
        # Correct answer validation
        # ----------------------------------------------------

        if correct not in [
            "A",
            "B",
            "C",
            "D"
        ]:
            continue

        # ----------------------------------------------------
        # Normalize options
        # ----------------------------------------------------

        normalized = {

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

        # ----------------------------------------------------
        # Make sure all options exist
        # ----------------------------------------------------

        if any(
            not normalized[key]
            for key in [
                "A",
                "B",
                "C",
                "D"
            ]
        ):

            continue

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        duplicate_key = (
            question
            .lower()
            .strip()
        )

        if duplicate_key in seen:
            continue

        seen.add(
            duplicate_key
        )

        # ----------------------------------------------------
        # Add valid question
        # ----------------------------------------------------

        valid.append({

            "question": question,

            "options": normalized,

            "correctAnswer": correct,

            "explanation": explanation

        })

    return valid


# ============================================================
# TRIM LARGE DOCUMENT
# ============================================================

def trim_document(
    text: str,
    max_chars: int = 120000
) -> str:

    """
    Prevent extremely large PDF text from creating
    an unnecessarily large Gemini request.

    Beginning, middle and end of the document are preserved.
    """

    if len(text) <= max_chars:

        return text

    third = max_chars // 3

    middle = len(text) // 2

    return (

        text[:third]

        + "\n\n"
        "[... middle section omitted for context size ...]"
        "\n\n"

        + text[
            middle - third // 2:
            middle + third // 2
        ]

        + "\n\n"
        "[... later section omitted for context size ...]"
        "\n\n"

        + text[-third:]
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def home():

    """
    Serve index.html from the project root.

    Project structure:

    project/
    │
    ├── api/
    │   └── index.py
    │
    └── index.html
    """

    index_path = (

        Path(__file__)
        .resolve()
        .parent
        .parent
        / "index.html"

    )

    if not index_path.exists():

        raise HTTPException(

            status_code=404,

            detail=(
                "index.html was not found. "
                "Make sure index.html is in the "
                "project root."
            )
        )

    return FileResponse(
        index_path
    )


# ============================================================
# API STATUS
# ============================================================

@app.get(
    "/api",
    include_in_schema=False
)
def api_root():

    return {

        "status": "ok",

        "message": (
            "Mock Paper Generator API "
            f"using {MODEL} is running."
        )

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/health"
)
def health():

    return {

        "status": "healthy",

        "model": MODEL

    }


# ============================================================
# GENERATE QUESTIONS
# ============================================================

@app.post(
    "/api/generate"
)
def generate_questions(
    request: GenerateRequest
):

    # --------------------------------------------------------
    # Get document text
    # --------------------------------------------------------

    document_text = (
        request.text
        .strip()
    )

    # --------------------------------------------------------
    # Validate document
    # --------------------------------------------------------

    if len(document_text) < 20:

        raise HTTPException(

            status_code=400,

            detail=(
                "The PDF does not contain "
                "enough readable text."
            )

        )

    # --------------------------------------------------------
    # Question count
    #
    # Default = 40
    # Maximum = 100
    # --------------------------------------------------------

    count = min(

        max(
            request.count,
            1
        ),

        100

    )

    # --------------------------------------------------------
    # Topic instruction
    # --------------------------------------------------------

    topic_instruction = ""

    if request.topic.strip():

        topic_instruction = f"""

FOCUS TOPIC:

{request.topic.strip()}

Generate questions ONLY about this topic.

If this topic is not present in the
source material, return an empty array.

"""


    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_instruction = f"""

You are an expert university examination
question writer.

Create a fresh mock examination from
the supplied study material.

Generate up to {count} high-quality
multiple-choice questions.

IMPORTANT:

The requested number of questions is {count}.

Try your best to generate the full
requested number when there is enough
usable information in the document.

{topic_instruction}

RULES:

1. Every question must be answerable
   from the supplied document.

2. Do not copy sentences directly
   from the source.

3. Write original question wording.

4. Test understanding, concepts,
   definitions, comparisons,
   processes, applications,
   examples and important facts.

5. Every question must have exactly
   four options.

6. Options must be A, B, C and D.

7. Exactly ONE option must be correct.

8. Incorrect options must be plausible.

9. Avoid ambiguous questions.

10. Do not repeat questions.

11. Do not closely rephrase
    another question.

12. Keep explanations concise
    and educational.

13. Do not invent facts that are
    not supported by the source.

14. Questions should have a good
    mixture of difficulty levels.

15. Avoid generating the same concept
    repeatedly unless the source contains
    multiple important aspects of it.

16. Return ONLY the structured
    JSON array.

"""


    # --------------------------------------------------------
    # Trim document if extremely large
    # --------------------------------------------------------

    document_text = trim_document(
        document_text
    )


    # --------------------------------------------------------
    # User prompt
    # --------------------------------------------------------

    prompt = f"""

SOURCE DOCUMENT:

{document_text}

END OF SOURCE DOCUMENT.

Generate up to {count} unique MCQs
based strictly on the source material.

"""


    # --------------------------------------------------------
    # Gemini generation
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(

            model=MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                system_instruction=
                    system_instruction,

                response_mime_type=
                    "application/json",

                response_schema=
                    QUESTION_SCHEMA,

                # Larger output because the user
                # can request up to 100 questions.
                max_output_tokens=30000

            )

        )

        # ----------------------------------------------------
        # Read Gemini response
        # ----------------------------------------------------

        raw = (
            response.text
            or ""
        ).strip()

        if not raw:

            raise ValueError(
                "Gemini returned an empty response."
            )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        parsed = json.loads(
            raw
        )

        # ----------------------------------------------------
        # Validate questions
        # ----------------------------------------------------

        questions = validate_questions(
            parsed
        )

        if not questions:

            raise ValueError(
                "Gemini did not generate "
                "any valid questions."
            )

        # ----------------------------------------------------
        # Never exceed requested count
        # ----------------------------------------------------

        questions = questions[:count]

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {

            "success": True,

            "requestedCount": count,

            "count": len(
                questions
            ),

            "questions": questions

        }

    except Exception as error:

        print(
            "\n"
            "========== GEMINI GENERATION ERROR =========="
        )

        print(
            repr(error)
        )

        print(
            "============================================="
            "\n"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Question generation failed: "
                + str(error)
            )

        )


# ============================================================
# AI DOUBT TUTOR
# ============================================================

@app.post(
    "/api/doubt"
)
def answer_doubt(
    request: DoubtRequest
):

    options = request.options

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Tutor prompt
    # --------------------------------------------------------

    prompt = f"""

You are a patient and knowledgeable
university tutor.

The student is reviewing this MCQ.

QUESTION:

{request.question}


OPTIONS:

A) {options.get("A", "")}

B) {options.get("B", "")}

C) {options.get("C", "")}

D) {options.get("D", "")}


CORRECT ANSWER:

{request.correctAnswer}
—
{options.get(
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

1. Stay focused on this question
   and its underlying concept.

2. Use simple language.

3. Correct misconceptions
   if necessary.

4. Refer to the options
   when useful.

5. Normally answer in
   2-5 sentences.

6. Do not contradict the
   supplied question context.

7. Explain the concept in a way
   that helps the student remember it.

"""


    # --------------------------------------------------------
    # Gemini tutor
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(

            model=MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                max_output_tokens=700

            )

        )

        answer = (
            response.text
            or ""
        ).strip()

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

        print(
            "\n"
            "========== GEMINI DOUBT ERROR =========="
        )

        print(
            repr(error)
        )

        print(
            "========================================"
            "\n"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Could not answer the doubt: "
                + str(error)
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
        )

    )
