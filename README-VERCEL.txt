VERCEL DEPLOYMENT

This project is already structured for Vercel.

Files:
- new.html          frontend
- api/index.py      FastAPI backend
- requirements.txt  Python dependencies
- vercel.json       Vercel configuration

1. Deploy with GitHub:
   - Upload this folder to a GitHub repository.
   - In Vercel, choose Add New -> Project.
   - Import the repository.
   - Deploy.

2. Add your Gemini key:
   Vercel Project -> Settings -> Environment Variables

   Name:
   GEMINI_API_KEY

   Value:
   your actual Gemini API key

3. Redeploy after adding the variable.

No Uvicorn start command is needed on Vercel.

CLI alternative:
   npm i -g vercel
   cd mock-paper-generator-vercel
   vercel
   vercel --prod

Local Vercel-style development:
   vercel dev

IMPORTANT:
Never commit .env or place the Gemini key inside new.html.
