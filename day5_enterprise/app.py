from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_pipeline import ask_question

app = FastAPI(title="Enterprise RAG API")

_ROOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Enterprise RAG API</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; padding: 0 1rem; }
    code { background: #f4f4f4; padding: 0.15rem 0.35rem; border-radius: 4px; }
    a { color: #0969da; }
  </style>
</head>
<body>
  <h1>Enterprise RAG API</h1>
  <p>This service is running. Open the docs to try requests, or call <code>POST /ask</code> from your app.</p>
  <ul>
    <li><a href="/docs">Swagger UI</a> — try <code>POST /ask</code> here</li>
    <li><a href="/redoc">ReDoc</a></li>
    <li><a href="/openapi.json">OpenAPI JSON</a></li>
  </ul>
  <p>Example body: <code>{"query": "What is Databricks?"}</code></p>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return _ROOT_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(req: QueryRequest):
    result = ask_question(req.query)
    return result