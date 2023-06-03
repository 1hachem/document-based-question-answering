from src.index import Index
from src import llms, splitters, embedding_models, measures

def bench_mark(contexts: list[str], questions: list[str], answers: list[str]) -> list[tuple]:
    assert len(contexts) == len(questions) == len(answers), "contexts, questions and answers must have the same length"
    inference = []
    for context, question, answer in zip(contexts, questions, answers):
        for splitter in splitters:
            for embedder in embedding_models:
                ## index document and embed query
                index = Index(embedder=embedder(), splitter=splitter())
                indexed_context = index(document=context)
                query_emb = index.embedder.embed([question])[0]
                
                for measure in measures:
                    candidates = measure().return_top(query_emb, indexed_context)
                    print(candidates)
                    sub_context = "\n\n".join(candidates)
                    prompt = f"""from this context : {sub_context}
                    answer this question : {question}
                    answer :"""
                    for llm in llms:
                        prediction = llm()(prompt)
                        ground_truth = answer
                        inference.append((prediction, ground_truth))
    return inference