RSMSSB Computer Instructor MCQ Platform - Persistent Version

Architecture:
- new.html: frontend
- api/index.py: FastAPI API
- MongoDB Atlas: permanent study metadata, questions, attempts
- MongoDB GridFS: original uploaded PDFs

Environment variables:
GEMINI_API_KEY=...
MONGODB_URI=...
MONGODB_DB=rsmssb_mcq
GEMINI_MODEL=gemini-3.6-flash

Vercel:
1. Put the project in GitHub.
2. Import into Vercel.
3. Add all environment variables in Project Settings -> Environment Variables.
4. Redeploy.

Important:
- Do not commit .env.
- MongoDB Atlas Network Access must allow the deployed server to connect.
- The upload endpoint uses multipart/form-data. Very large PDFs may exceed a hosting provider's request-size limit; for large-file production deployments, use object storage/direct browser uploads.


