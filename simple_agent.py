import os
import time
import random
import hashlib
import inspect
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
from openai import OpenAI
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import from local files
import tools
from prompts import system_prompt # Import the prompt from the .py file


# --- Environment Setup ---
# pip install openai langchain langchain-openai faiss-cpu tiktoken numpy

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
if os.environ["OPENAI_API_KEY"] == "YOUR_API_KEY_HERE":
    print("Warning: OPENAI_API_KEY is not set. The script will not run correctly.")

# Initialize OpenAI client
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini" # Updated to gpt-4o-mini
EMBEDDINGS_DIR = "embeddings"

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
    embeddings_cache_file, chunks_cache_file = f"{cache_prefix}_embeddings.npy", f"{cache_prefix}_chunks.txt"

    if os.path.exists(embeddings_cache_file) and os.path.exists(chunks_cache_file):
        print("--- Loading embeddings and chunks from cache ---")
        embeddings = np.load(embeddings_cache_file)
        with open(chunks_cache_file, 'r', encoding='utf-8') as f:
            TEXT_CHUNKS = f.read().split('¶')
    else:
        print("--- No cache found. Generating new embeddings... ---")
        raw_chunks = split_into_chunks(novel_text, chunk_size=chunk_size)
        TEXT_CHUNKS = [f"{i}: {chunk}" for i, chunk in enumerate(raw_chunks)]
        embeddings = embed_chunks_parallel(TEXT_CHUNKS, chunk_size)
        np.save(embeddings_cache_file, embeddings)
        with open(chunks_cache_file, 'w', encoding='utf-8') as f:
            f.write('¶'.join(TEXT_CHUNKS))

    embedding_dim = embeddings.shape[1]
    FAISS_INDEX = faiss.IndexFlatL2(embedding_dim)
    FAISS_INDEX.add(np.array(embeddings).astype('float32'))
    print("--- RAG Pipeline is READY ---")

# --- Agent Definition ---

def create_novel_agent():
    """
    Creates the JSON Chat agent for answering questions about a novel.
    """
    # Automatically detect and load tools from the tools.py module
    agent_tools = [
        obj for _, obj in inspect.getmembers(tools) if isinstance(obj, BaseTool)
    ]
    
    # The system_prompt is now imported directly
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Use a Chat model as required by the JSON chat agent
    llm = ChatOpenAI(model=LLM_MODEL)

    # Create the agent using the modern create_json_chat_agent constructor
    agent = create_json_chat_agent(llm, agent_tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=agent_tools, 
        verbose=True,
        handle_parsing_errors=True # Helpful for debugging JSON output issues
    )

    return agent_executor

if __name__ == '__main__':
    # --- Setup and Run ---
    novel_path = "assets/1.txt"
    setup_rag_pipeline(novel_path)

    # Pass necessary data to the tools module after it's loaded
    tools.TEXT_CHUNKS = TEXT_CHUNKS
    tools.LLM_MODEL = LLM_MODEL
    
    novel_agent_executor = create_novel_agent()
    question = "What happens after the third time '当前百世书残留页数' is mentioned?"
    
    print("\nAgent created and pipeline is ready. Running query...")
    counter = 0
    for step in novel_agent_executor.stream({"input": question}):
        counter += 1
        print(f"\n--- Step {counter}---")
        step["messages"][-1].pretty_print()
        if "Final Answer" in str(step["messages"][-1].content[:15]):
            break
    
    print("\n--- Final Result ---")
    step["messages"][-1].pretty_print()

