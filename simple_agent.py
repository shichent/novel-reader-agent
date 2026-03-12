import os
import inspect
from typing import List

import faiss
import numpy as np
from openai import OpenAI
from langchain.tools import BaseTool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Import from local files
import tools
import rag
from prompts import training_example1, system_prompt


# --- Environment Setup ---
# pip install openai langchain langchain-openai faiss-cpu tiktoken numpy

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
if os.environ["OPENAI_API_KEY"] == "YOUR_API_KEY_HERE":
    print("Warning: OPENAI_API_KEY is not set. The script will not run correctly.")

# Initialize OpenAI client
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5"
EMBEDDINGS_DIR = "embeddings"


# --- Agent Definition ---

def create_novel_agent(graph_func):
    # Auto-discover all BaseTool instances from tools module (excludes factory functions)
    agent_tools = [
        obj for _, obj in inspect.getmembers(tools) if isinstance(obj, BaseTool)
    ]
    
    # Add the graphrag tool with graph_func injected via closure
    agent_tools.append(tools.make_graphrag_tool(graph_func))

    llm = ChatOpenAI(model=LLM_MODEL)
    agent = create_agent(
        llm,
        tools=agent_tools,
        system_prompt=system_prompt + training_example1,
    )
    return agent


if __name__ == '__main__':
    # --- Setup and Run ---
    working_dir = "assets/book1/"
    novel_name = "1.txt"
    question = "What happens after the third time '当前百世书残留页数' is mentioned?"
    verbose = True

    rag.setup_rag_pipeline(working_dir + novel_name)
    graph_func = rag.setup_graphrag(file_name=novel_name, working_dir=working_dir)

    # Pass necessary data to the tools module after it's loaded
    tools.TEXT_CHUNKS = rag.TEXT_CHUNKS
    tools.LLM_MODEL = LLM_MODEL
    tools.EMBEDDING_MODEL = EMBEDDING_MODEL
    tools.FAISS_INDEX = rag.FAISS_INDEX

    novel_agent = create_novel_agent(graph_func)

    print("\nAgent created and pipeline is ready. Running query...")
    counter = 0

    # LangChain 1.0: stream() yields message chunks directly from the agent
    for chunk in novel_agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                counter += 1
                if verbose:
                    print(f"\n--- Step {counter} [{node_name}] ---")
                    if hasattr(msg, "pretty_print"):
                        msg.pretty_print()

    # After the loop, the last message IS the final answer
    print("\n--- Final Result ---")
    last_msg = messages[-1]  # last message from last chunk
    if hasattr(last_msg, "pretty_print"):
        last_msg.pretty_print()