
training_example1 = """
Training example 1: Question: What happened after the second time 'magic sword' was mentioned?

Thought: The user is asking about events that occurred *after* the second mention of the phrase "magic sword".
1. I need to find all occurrences of "magic sword" to identify the second one.
2. Once I have the list of positions, I need to select the second position.
3. Then, I need to find the positions of the chunks that come *after* that second position.
4. After that, I need to retrieve the text chunks at those positions to understand what happened next.
5. Finally, I'll answer the question based on the retrieved text.
Action:
```json
{{
  "thought": "I need to find all occurrences of the keyword 'magic sword' in the text to identify the second mention.",
  "action": "find_position_by_keyword",
  "action_input": {{
    "keyword": "magic sword"
  }}
}}
```
Observation: [1, 3, 5, 6]
Action:
```json
{
  "thought": "I have found the positions of all mentions of 'magic sword'. The second mention is at index 3. Now, I need to select the positions that come after this index, which are positions 3,4,5 and 6, to find out what happened next.",
  "action": "retrieve_text_chunks",
  "action_input": {{
    "positions": [3, 4, 5, 6],
  }}
}}
```
Observation: 3: The hero wielded the magic sword to vanquish the dragon. 4: With the dragon defeated, peace returned to the kingdom. 5: The magic sword was then placed in a shrine for safekeeping. 6: Legends of the magic sword spread far and wide.
Action:
```json
{
  "thought": "Based on the retrieved context, after the second mention of 'magic sword', the hero used it to defeat the dragon, peace was restored, it was placed in a shrine, and its legend spread. I can now provide the final answer.",
  "action": "final_answer",
  "action_input": {{
    "answer": "After the second mention of 'magic sword', the hero used it to vanquish the dragon, peace returned to the kingdom, the sword was placed in a shrine for safekeeping, and legends of the magic sword spread far and wide."
  }}
}}
"""

system_prompt = """You are an agent designed to answer questions about a novel by reading it.
You have access to a set of tools. You must format your thoughts and actions as a JSON blob with 'thought', 'action' and 'action_input' keys.

You have access to the following tools:
{tools}

Use the following format:

Action:
```json
{{
  "thought": "fully explain your thought that supports your choice of action",
  "action": "the action to take, should be one of [{tool_names}]",
  "action_input": "the input to the action"
}}
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times. When you have enough information to answer the question, you must use the following format to provide the final answer.)
Action:
```json
{{
  "thought": "explain why you have enough information to provide a final answer, and explain how you arrived at the final answer",
  "action": "final_answer",
  "action_input": "provide your final answer here"
}}

Here are some training examples: 
"""