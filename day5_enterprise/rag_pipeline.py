"""Hybrid BM25 + NumPy L2 RAG pipeline (same logic as rag_pipeline.ipynb, API-shaped)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

for _root in [Path.cwd(), *Path.cwd().resolve().parents]:
    _env = _root / ".env"
    if _env.is_file():
        load_dotenv(_env)
        break

documents = [
    "Databricks is a unified analytics platform for big data and AI.",
    "LangChain helps build applications using large language models.",
    "FastAPI is a modern framework for building APIs with Python.",
    "Hybrid search combines keyword and semantic search techniques.",
    "BM25 is a ranking algorithm used in search engines.",
    "Vector databases store embeddings for semantic retrieval.",
]

llm = ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
embeddings = OpenAIEmbeddings()

tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

doc_matrix = np.array(
    [embeddings.embed_query(doc) for doc in documents], dtype=np.float32
)

_cache: dict[str, dict] = {}


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


def rrf_fusion(bm25_rank, vector_rank, k: int = 60):
    scores = {}
    for rank, idx in enumerate(bm25_rank):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
    for rank, idx in enumerate(vector_rank):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rewrite_query(query: str) -> str:
    prompt = f"Rewrite this query clearly for better search: {query}"
    return llm.invoke([HumanMessage(content=prompt)]).content


def ask_question(user_query: str) -> dict:
    """Return answer, retrieved chunks, and metadata for the HTTP API / Streamlit UI."""
    key = _cache_key(user_query)
    if key in _cache:
        hit = dict(_cache[key])
        hit["source"] = "cache"
        return hit

    rewritten = rewrite_query(user_query)

    tokenized_query = rewritten.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_rank = np.argsort(bm25_scores)[::-1]

    q = np.asarray(embeddings.embed_query(rewritten), dtype=np.float32)
    dists = np.linalg.norm(doc_matrix - q, axis=1)
    vector_rank = np.argsort(dists)[:3]

    fused = rrf_fusion(bm25_rank[:3], vector_rank)
    top_docs = [documents[idx] for idx, _ in fused[:3]]

    context = "\n".join(top_docs)
    final_prompt = f"""
    Answer based on context only:

    Context:
    {context}

    Question:
    {user_query}
    """
    answer = llm.invoke([HumanMessage(content=final_prompt)]).content

    relevance_prompt = f"""
    Is the answer relevant to the question?

    Question: {user_query}
    Answer: {answer}

    Answer Yes or No.
    """
    relevance = llm.invoke([HumanMessage(content=relevance_prompt)]).content

    out = {
        "answer": answer,
        "docs": top_docs,
        "relevance": relevance,
        "source": "hybrid_bm25_numpy_l2_rrf",
    }
    _cache[key] = out
    return out
