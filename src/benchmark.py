from src import embedding_models, llms, measures, splitters
from src.evaluation import is_correct
from src.index import Index

BATCH_SIZE = 4


def construct_indexes(
    contexts: list[str], questions: list[str], answers: list[str]
) -> list[tuple]:
    assert (
        len(contexts) == len(questions) == len(answers)
    ), "contexts, questions and answers must have the same length"

    indexes = []
    for splitter in splitters:
        for embedder in embedding_models:
            for measure in measures:
                index = Index(embedder=embedder(), splitter=splitter())
                triplets = []
                for context, question, answer in zip(contexts, questions, answers):
                    indexed_context = index(document=context)
                    query_emb = index.embedder.embed([question])[0]

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
                        "measure": measure().__class__.__name__,
                        "triplets": triplets,
                    }
                )

    return indexes


def inference(indexes: list[dict]) -> list[str]:
    prompt = (
        lambda context, question: f"Answer the question only from the given context, Context: {context}\nQuestion: {question}\nAnswer:"
    )
    answers = []
    requests = []
    for index in indexes:
        for triplet in index["triplets"]:
            requests.append(prompt(triplet["sub_context"], triplet["question"]))
            answers.append(triplet["answer"])
            triplet["predictions"] = {}

    for llm in llms:
        llm = llm()

        predictions = []
        for i in range(0, len(requests), BATCH_SIZE):
            batch = requests[i : i + BATCH_SIZE]
            predictions += llm(batch)

        for index in indexes:
            for triplet in index["triplets"]:
                triplet["predictions"][llm.__class__.__name__] = {
                    "answer": predictions.pop(0)
                }

    return indexes


async def evaluate_inferences(indexes: list[dict]) -> list[dict]:
    for index in indexes:
        for triplet in index["triplets"]:
            for llm in llms:
                triplet["predictions"][llm().__class__.__name__][
                    "evaluation"
                ] = await is_correct(
                    triplet["answer"],
                    triplet["predictions"][llm().__class__.__name__]["answer"],
                )

    return indexes


def parse_results(indexes: list[dict]):
    table = []
    llms_names = indexes[0]["triplets"][0]["predictions"].keys()
    l = len(indexes[0]["triplets"])

    for index in indexes:
        row = {
            "text_splitter": index["text_splitter"],
            "embedder": index["embedder"],
            "measure": index["measure"],
        }
        for llm in llms_names:
            row[llm] = 0

        table.append(row)

        for llm in llms_names:
            for triplet in index["triplets"]:
                table[-1][llm] += triplet["predictions"][llm]["evaluation"]
            table[-1][llm] /= l

    return table
