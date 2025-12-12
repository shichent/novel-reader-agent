import os
import time
import random
import hashlib
import inspect
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import sqlite3
from nano_graphrag import GraphRAG, QueryParam

import faiss
from openai import OpenAI

client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDINGS_DIR = "embeddings"
DB_FILE = "embeddings/chunks.db"

# --- RAG Pipeline Components ---
TEXT_CHUNKS = []
FAISS_INDEX = None

def split_into_chunks(text, chunk_size=500,overlap=0,delimiter="\n"):
    chunks = []
    p = 0
    q = 0
    while q < len(text):
        q = text.find(delimiter,p+chunk_size)
        if q == -1:
            q = len(text)
        chunks.append(text[p:q])
        pp = text.rfind(delimiter,0,max(q-overlap,0))
        if pp <= p:
            p = q
        else:
            p = pp
    return chunks

def get_embedding_batch(text_batch, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text_batch)
            return [d.embedding for d in resp.data]
        except Exception as e:
            time.sleep(2 ** attempt + random.random())
    raise RuntimeError("Max retries exceeded for embedding batch.")

def embed_chunks_parallel(text_chunks, chunk_size, max_workers=8):
    batch_size = int(8192 / (chunk_size * 1.2))
    batches = [text_chunks[i:i + batch_size] for i in range(0, len(text_chunks), batch_size)]
    embeddings = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_embedding_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            embeddings.extend(future.result())
    return np.array(embeddings)

def setup_rag_pipeline(file_path: str, chunk_size: int = 500):
    global TEXT_CHUNKS, FAISS_INDEX
    print(f"--- Setting up RAG pipeline for: {file_path} ---")
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    with open(file_path, 'r', encoding='utf-8') as f: novel_text = f.read()
    file_hash = hashlib.md5(novel_text.encode()).hexdigest()[:8]
    cache_prefix = os.path.join(EMBEDDINGS_DIR, f"{os.path.basename(file_path)}_{file_hash}_cs{chunk_size}")
    embeddings_cache_file = f"{cache_prefix}_embeddings.npy"
    doc_hash_size = f"{file_hash}_cs{chunk_size}"

    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, value TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS chunks (chunk_id INTEGER, doc_id INTEGER, chunk TEXT, PRIMARY KEY (chunk_id, doc_id))")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM documents where value='{doc_hash_size}' LIMIT 1")
    cached_doc_id = cursor.fetchone()
    cached_doc_id = cached_doc_id[0] if cached_doc_id is not None else None

    if os.path.exists(embeddings_cache_file) and cached_doc_id:
        print("--- Loading embeddings and chunks from cache ---")
        embeddings = np.load(embeddings_cache_file)
        cursor.execute(f"SELECT chunk FROM chunks WHERE doc_id = {cached_doc_id} ORDER BY chunk_id")
        TEXT_CHUNKS = [row[0] for row in cursor.fetchall()]
    else:
        print("--- No cache found. Generating new embeddings... ---")
        raw_chunks = split_into_chunks(novel_text, chunk_size=chunk_size)
        TEXT_CHUNKS = [f"{i}: {chunk}" for i, chunk in enumerate(raw_chunks)]
        embeddings = embed_chunks_parallel(TEXT_CHUNKS, chunk_size)
        np.save(embeddings_cache_file, embeddings)
        doc_id = cursor.execute("INSERT INTO documents (value) VALUES (?)", (doc_hash_size,)).lastrowid
        data_to_insert = [(i, doc_id, chunk) for i, chunk in enumerate(TEXT_CHUNKS)]
        with conn:
            cursor.executemany("INSERT INTO chunks (chunk_id, doc_id, chunk) VALUES (?, ?, ?)", data_to_insert)
        conn.commit()
        conn.close()

    embedding_dim = embeddings.shape[1]
    FAISS_INDEX = faiss.IndexFlatL2(embedding_dim)
    FAISS_INDEX.add(np.array(embeddings).astype('float32'))
    print("--- RAG Pipeline is READY ---")

def setup_graphrag(file_name:str,working_dir:str):
    graph_func = GraphRAG(working_dir=working_dir)

    with open(working_dir+file_name) as f:
        graph_func.insert(f.read())
    return graph_func