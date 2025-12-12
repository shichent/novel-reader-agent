from nano_graphrag import GraphRAG, QueryParam
import asyncio
import json
import random
import time
import logging

# Set the logging level to CRITICAL (only the most severe errors show)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

from nano_graphrag.MinHash import HashTable, MinHash, jaccard_similarity
from nano_graphrag._llm import gpt_4o_complete, gpt_5_1_complete

stats = {}
for bookid in [1,2]:

    graph_func = GraphRAG(working_dir=f"assets/book{bookid}")

    # Insert the novel text into the graphrag
    # with open(f"assets/book{bookid}/{bookid}.txt",encoding="utf-8") as f:
    #     graph_func.insert(f.read())

    # # Access the underlying NetworkX graph object
    # internal_graph = graph_func.chunk_entity_relation_graph._graph

    # # 1. Print All Nodes (with attributes like description, entity type)
    # print(f"Total Nodes: {internal_graph.number_of_nodes()}")
    # counter = 0
    # print("\"陈国\"")
    # print("陈国")
    # for node, data in internal_graph.nodes(data=True):
    #     counter += 1
    #     if counter > 3:
    #         break
    #     print(f"Data type: {type(node)}")
    #     print(f"  Type: {data.get('type', 'Unknown')}")
    #     print(f"  Description: {data.get('description', 'No description')[:50]}...") # Truncated for display
    #     print(f"  Node ID: {}")

    # # 2. Print All Edges (with attributes like weight, description)
    # print(f"\nTotal Edges: {internal_graph.number_of_edges()}")
    # for source, target, data in internal_graph.edges(data=True):
    #     counter += 1
    #     if counter > 6:
    #         break
    #     print(f"Edge: {source} -> {target}")
    #     print(f"  Weight: {data.get('weight', 0)}")
    #     print(f"  Description: {data.get('description', 'No description')[:50]}...")

    # Perform global graphrag search
    # print(graph_func.query("What are the top themes in this story?"))

    # query_text = "陈国与初圣宗之间维持着一种什么样的交换关系？"

    # hashcodes = MinHash(query_text,graph_func.hashtable.K*graph_func.hashtable.L)
    # results = graph_func.hashtable.lookup(hashcodes)
    # print(f"number of reslts: {len(results)}")
    # results_scored = []
    # for r in results:
    #     sim = jaccard_similarity(r,query_text,1)
    #     results_scored.append((sim,r))
    # results_scored.sort(reverse=True)
    # results = [{
    #     'entity_name': r
    # } for s,r in results_scored[:20]]
    # print("Candidates from LSH:", results)


    # Perform local graphrag search (I think is better and more scalable one)
    # ans = graph_func.query(query_text, param=QueryParam(mode="local",only_need_context=False,use_LSH=True))
    # print(ans)
    # results = asyncio.run(graph_func.entities_vdb.query("这本小说的主角是谁？", top_k=20))
    # print(results[0])

    def permute(question):
        new_order = ['A','B','C','D']
        while new_order[ord(question['answer']) - ord('A')] == question['answer']:
            random.shuffle(new_order)
        permuted_question = {
            'question': question['question'],
            'A': question[new_order[0]],
            'B': question[new_order[1]],
            'C': question[new_order[2]],
            'D': question[new_order[3]],
            'answer': chr(ord('A') + new_order.index(question['answer']))
        }
        return permuted_question


    async def ask_one_question(question,use_LSH):
        output = {}
        ans = await graph_func.aquery(question['question'],param=QueryParam(mode="local",use_LSH=use_LSH))
        # ans = ""
        system_prompt = "You are a helpful agent who can answer questions based on the provided context. You will be provided a question and 4 choices, labelled A,B,C,D. Choose the most appropriate option (A,B,C,D) based on the provided context. Do not use internet search, do not use any other knowledges. Your answer should start with one of the options (A,B,C or D), then provide a brief explanation about how the context support that choice."
        choice = await gpt_4o_complete(f"Question: {question['question']}\nA: {question['A']}\nB: {question['B']}\nC: {question['C']}\nD: {question['D']}\nContext: {ans}\n",system_prompt=system_prompt,max_tokens=50)
        return choice

    async def ask_all_questions(questions,use_LSH):
        tasks = []
        for q in questions:
            tasks.append(ask_one_question(q,use_LSH=use_LSH))
        results = await asyncio.gather(*tasks)
        return results

    with open(f"C:\\Users\\emper\\OneDrive\\Code\\novel-reader-agent\\assets\\book{bookid}\\test_questions_{bookid}.json","r",encoding='utf-8') as f:
        questions = json.load(f)
    # q = questions[1]
    # q = permute(q)
    # results = asyncio.run(ask_one_question(q))
    # print("result: "+results)
    # print(q)
    # print("Context:")
    # context = graph_func.query(q['question'],param=QueryParam(mode="local",use_LSH=True))
    # print(context)
    print(f"Total questions: {len(questions)}")
    correct = 0
    questions_batch = [questions[:30],questions[30:60],questions[60:]]
    for qs in questions_batch:
        for i in range(len(qs)):
            qs[i] = permute(qs[i])
        results = asyncio.run(ask_all_questions(qs,use_LSH=True))
        for r,q in zip(results,qs):
            if r[0].lower() == q['answer'].lower():
                correct += 1
        time.sleep(120)
    stats[bookid] = correct
total_questions = {1:96,2:113}
for bookid in [1,2]:
    correct = stats[bookid]
    print(f"correct: {correct}")
    print(f"accuracy: {correct/total_questions[bookid]*100:.2f}%")
print(f"total correct: {sum(stats.values())}")
print(f"total accuracy: {sum(stats.values())/(96+113)*100:.2f}%")
print(f"K:{graph_func.hashtable.K}, L:{graph_func.hashtable.L}, B:{graph_func.hashtable.B}, R:{graph_func.hashtable.R}")
print(f"Top k: {QueryParam.top_k}")