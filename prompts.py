
training_example1 = """
Training example 1: Question: What happened after the second time 'magic sword' was mentioned?

Thought: The user is asking about events that occurred *after* the second mention of the phrase "magic sword".
1. I need to find all occurrences of "magic sword" to identify the second one.
2. Once I have the list of positions, I need to select the second position.
3. Then, I need to find the positions of the chunks that come *after* that second position.
4. I need to remove the sequential and counting keywords from the question because sequential and counting are already handled by finding positions. This modified question should be in the same language as original question.
5. Finally, I'll use that text as context to answer the modified question.
Action:
```json
{{
  "action": "find_position_by_keyword",
  "action_input": {{
    "keyword": "magic sword"
  }}
}}
```
Observation: [1, 3]
Thought: The keyword "magic sword" appears at positions 1 and 3. The user is interested in what happens after the *second* mention, which is at position 3 (index 1 in the list). Now I need to get the chunks *after* position 3. The `manipulate_position_list` tool can do this. I'll use `select_after_index_n:1`.
Action:
```json
{{
  "action": "manipulate_position_list",
  "action_input": {{
    "positions": [1, 3],
    "action": "select_after_index_n:1"
  }}
}}
```
Observation: [3, 4, 5, 6]
Thought: Now I have the positions of the chunks after the second mention of "magic sword". To answer the original question, I need to answer the question "What has happened in the context?" based on the context from these positions. I need to make sure this modified question is in the same language as original question. I'll use the `answer_question` tool for this.
Action:
```json
{{
  "action": "answer_question",
  "action_input": {{
    "positions": [3, 4, 5, 6],
    "question": "What has happened in the context?"
  }}
}}
```
Observation: Final Answer: The knight realized the magic was his own courage and faced the dragon again with confidence.
Final Answer: The knight realized the magic was his own courage and faced the dragon again with newfound confidence."""



system_prompt = """You are an agent designed to answer questions about a novel by reading it.
You have access to a set of tools. You must format your thoughts and actions as a JSON blob with 'action' and 'action_input' keys.

You have access to the following tools:
{tools}

Use the following format:

Thought: you should always think about what to do
Action:
```json
{{
  "action": "the action to take, should be one of [{tool_names}]",
  "action_input": "the input to the action"
}}
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here are some training examples: 
"""