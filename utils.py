from openai import OpenAI
import PyPDF2
import faiss
import os
import numpy as np
from typing import List
import tiktoken

# Instancia o cliente com chave da API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text(text: str, max_tokens=500) -> List[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(tokenizer.encode(" ".join(current))) > max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))
    return chunks

def embed_chunks(chunks: List[str]) -> tuple:
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        vector = response.data[0].embedding
        embeddings.append(vector)
    return np.array(embeddings).astype("float32"), chunks

def search_similar(query: str, index, chunks: List[str], top_k=3) -> List[str]:
    query_embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    D, I = index.search(np.array([query_embed], dtype='float32'), top_k)
    return [chunks[i] for i in I[0]]
