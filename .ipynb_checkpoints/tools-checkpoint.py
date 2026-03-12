from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
import numpy as np

from nano_graphrag import GraphRAG, QueryParam
import rag

# Initialize the OpenAI client for use in this module
client = OpenAI()

# This list will be populated when setup_rag_pipeline is called from the main script.
TEXT_CHUNKS = []
LLM_MODEL = "gpt-5"
EMBEDDING_MODEL = "text-embedding-3-small"
FAISS_INDEX = None

@tool
def find_position_by_keyword(keyword: str) -> list[int]:
    """
    Finds all chunk positions that contain an exact keyword.
    Returns a list of integer positions.
    """
    positions = []
    for i, chunk in enumerate(TEXT_CHUNKS):
        if keyword in chunk:
            positions.append(i)
    return positions

@tool
def simple_rag_search(query: str, top_k: int = 5) -> str:
    """
    Performs a simple RAG search to find the top_k most relevant text chunks for the query. Use this tools to find relevant exact keywords related to an ambiguous query.
    Returns concatenated text of those chunks.
    """
    if FAISS_INDEX is None:
        return "FAISS index not initialized."

    # Get embedding for the query
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_embedding = resp.data[0].embedding

    # Search in FAISS index
    D, I = FAISS_INDEX.search(np.array([query_embedding]), top_k)
    
    # Retrieve and concatenate the relevant text chunks
    results = []
    for idx in I[0]:
        if 0 <= idx < len(TEXT_CHUNKS):
            results.append(f"{idx}: {TEXT_CHUNKS[idx]}")
    
    return " ".join(results)

# @tool
# def manipulate_position_list(positions: list[int], action: str) -> list[int]:
#     """
#     Filters or selects positions from a list.
#     Valid actions:
#     - 'select_index_n:<n>' (e.g., 'select_index_n:1' for the second element)
#     - 'select_after_index_n:<n>' (e.g., 'select_after_index_n:1')
#     """
#     parts = action.split(':')
#     op = parts[0]
#     if op == 'select_index_n' and len(parts) > 1:
#         try:
#             index = int(parts[1])
#             if 0 <= index < len(positions):
#                 return [positions[index]]
#             else:
#                 return []
#         except (ValueError, IndexError):
#             return []
#     elif op == 'select_after_index_n' and len(parts) > 1:
#         try:
#             index = int(parts[1])
#             if 0 <= index < len(positions):
#                 start_pos = positions[index]
#                 # Assuming we want the next few chunks
#                 return list(range(start_pos, min(start_pos + 10, len(TEXT_CHUNKS))))
#             else:
#                 return []
#         except (ValueError, IndexError):
#             return []
#     return []

@tool
def retrieve_text_chunks(positions: list[int]) -> str:
    """
    Retrieves text chunks at the specified positions.
    Returns concatenated text of those chunks and their indices.
    """
    ans = ''
    for pos in positions:
        if 0 <= pos < len(TEXT_CHUNKS):
            ans += f"{pos}: {TEXT_CHUNKS[pos]} "
    return ans.strip()

@tool
def answer_question_with_graphrag(question: str, graph_func: GraphRAG) -> str:
    """
    Uses GraphRAG to answer a question based on the inserted text data.
    """
    query_param = QueryParam(query=question, top_k=5)
    response = graph_func.query(query_param)
    return response

@tool
def final_answer(answer: str) -> str:
    """
    Returns the final answer.
    """
    return answer