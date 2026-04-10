import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("🎬 Enterprise RAG Assistant")

query = st.text_input("Ask a question")

if st.button("Submit"):
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"query": query},
        timeout=120,
    )

    data = response.json()

    st.write("### ✅ Answer")
    st.write(data["answer"])

    st.write("### 📄 Retrieved Docs")
    for doc in data["docs"]:
        st.write("-", doc)

    st.write("### ⚡ Source:", data["source"])