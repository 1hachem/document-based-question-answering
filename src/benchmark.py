from src import embedding_models, llms, measures, splitters
from src.index import Index


def construct_indexes(
    contexts: list[str], questions: list[str], answers: list[str]
) -> list[tuple]:
    assert (
        len(contexts) == len(questions) == len(answers)
    ), "contexts, questions and answers must have the same length"

    indexes = []
    for context, question, answer in zip(contexts, questions, answers):
        for splitter in splitters:
            for embedder in embedding_models:
                ## index document and embed query
                index = Index(embedder=embedder(), splitter=splitter())
                indexed_context = index(document=context)
                query_emb = index.embedder.embed([question])[0]

                triplets = []
                for measure in measures:
                    candidates = measure().return_top(query_emb, indexed_context)
                    sub_context = "\n\n".join(candidates)
                    triplet = {
                        "question": question,
                        "sub_context": sub_context,
                        "answer": answer,
                    }
                    triplets.append(triplet)

                indexes.append(
                    {
                        "text_splitter": splitter().__class__.__name__,
                        "embedder": embedder().__class__.__name__,
                        "triplets": triplets,
                    }
                )

    return indexes


def inference(indexes: dict) -> list[str]:
    prompt = (
        lambda context, question: f"Answer the question only from the given context, Context: {context}\nQuestion: {question}\nAnswer:"
    )
    answers = []
    requests = []
    for index in indexes:
        for triplet in index["triplets"]:
            requests.append(prompt(triplet["sub_context"], triplet["question"]))
            answers.append(triplet["answer"])

    for llm in llms:
        llm = llm()
        predictions = llm(requests)

        for index in indexes:
            for triplet in index["triplets"]:
                triplet[llm.__class__.__name__] = predictions.pop(0)

    return indexes
