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
import rag
from prompts import training_example1,system_prompt # Import the prompt from the .py file


# --- Environment Setup ---
# pip install openai langchain langchain-openai faiss-cpu tiktoken numpy

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
if os.environ["OPENAI_API_KEY"] == "YOUR_API_KEY_HERE":
    print("Warning: OPENAI_API_KEY is not set. The script will not run correctly.")

# Initialize OpenAI client
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5" # Updated to gpt-4o-mini
EMBEDDINGS_DIR = "embeddings"



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
        ("system", system_prompt+training_example1),
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
    novel_name = "1.txt"
    question = "What happens after the third time '当前百世书残留页数' is mentioned?"
    verbose = True
    rag.setup_rag_pipeline("assets/"+novel_name)

    # Pass necessary data to the tools module after it's loaded
    tools.TEXT_CHUNKS = rag.TEXT_CHUNKS
    tools.LLM_MODEL = LLM_MODEL
    
    novel_agent_executor = create_novel_agent()
    
    print("\nAgent created and pipeline is ready. Running query...")
    counter = 0
    for step in novel_agent_executor.stream({"input": question}):
        counter += 1
        if "Final Answer" in str(step["messages"][-1].content[:15]):
            break
        if verbose:
            print(f"\n--- Step {counter}---")
            step["messages"][-1].pretty_print()
    
    print("\n--- Final Result ---")
    step["messages"][-1].pretty_print()

