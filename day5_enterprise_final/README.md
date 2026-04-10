# Enterprise RAG Agent Demo

This repo is a local-first question-answering demo:

- It searches local knowledge in `data/` first.
- If the answer is not in local documents, it uses a free web-search tool.
- The UI shows whether the final answer came from local files, the web, or both.

## Quick Start

```bash
cp .env.example .env
# add your OpenAI API key to .env
./start_demo.sh
```

Open:

- UI: `http://127.0.0.1:8501`
- API: `http://127.0.0.1:8000`

If port `8000` is already in use:

```bash
BACKEND_PORT=8001 ./start_demo.sh
```

## Manual Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your OpenAI API key to .env
uvicorn app:app --reload --port 8000
BACKEND_URL=http://127.0.0.1:8000 streamlit run ui.py --server.port 8501
```

## Demo Questions

Local knowledge questions:

- `What is Orbit?`
- `Which teams use Orbit?`
- `What are the support hours?`

Questions that should trigger the web tool:

- `Who is the CEO of OpenAI?`
- `What happened in AI this week?`

## Add your own data

Place any of these file types under `data/`:

- `.pdf`
- `.txt`
- `.md`

If `data/` is empty, the app uses a built-in demo corpus so the project still works out of the box.

## Useful endpoints

- `GET /health`
- `GET /status`
- `POST /ask`

Example request body:

```json
{
  "query": "What is Orbit?"
}
```

## Repo Notes

- `data/` contains dummy local documents for the demo.
- `start_demo.sh` starts both the backend and Streamlit UI.
- `.env` is ignored locally; `.env.example` is the template to copy.
