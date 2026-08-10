// Node.js Vercel Function (Web-standard fetch handler — no framework needed).
// Responsibility: issue short-lived client tokens so the browser can upload
// PDFs (and their extracted text) DIRECTLY to Vercel Blob, never through
// this function's own request body. This is what avoids both:
//   - Starlette/FastAPI's 1MB multipart part-size limit
//   - Vercel's hard 4.5MB serverless function body limit
//
// Requires the "@vercel/blob" package (see package.json) and a Blob store
// connected to this project (Vercel Dashboard -> Storage -> Create Blob
// store -> Connect to this project). That step adds BLOB_READ_WRITE_TOKEN
// to your environment variables automatically.

const { handleUpload } = require('@vercel/blob/client');

const MAX_PDF_BYTES = 220 * 1024 * 1024; // ~220MB headroom over 200MB PDFs
const MAX_TEXT_BYTES = 50 * 1024 * 1024; // extracted text is always far smaller than the PDF

module.exports = {
  async fetch(request) {
    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 });
    }

    const body = await request.json();

    try {
      const jsonResponse = await handleUpload({
        body,
        request,
        onBeforeGenerateToken: async (pathname) => {
          // pathname is whatever the client passes to upload()/uploadPresigned().
          // We key the allowed size/type off a prefix so one route can serve
          // both the PDF and its extracted-text sidecar.
          const isText = pathname.startsWith('study-text/');
          return {
            allowedContentTypes: isText
              ? ['text/plain']
              : ['application/pdf'],
            maximumSizeInBytes: isText ? MAX_TEXT_BYTES : MAX_PDF_BYTES,
            addRandomSuffix: true,
          };
        },
        onUploadCompleted: async () => {
          // No DB write here — the client tells /api/studies about the
          // finished blob URL(s) itself once both uploads complete.
        },
      });

      return Response.json(jsonResponse);
    } catch (error) {
      return Response.json(
        { error: error instanceof Error ? error.message : String(error) },
        { status: 400 },
      );
    }
  },
};
