import asyncio
import os

import dotenv
import lmql
import openai

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@lmql.query
async def evaluate_(prediction: str, ground_truth: str):
    '''argmax
        """output 1 if the student answer is the same as true answer, 0 otherwise
        ignore case and small differences
        student answer: {prediction}
        true answer: {ground_truth}
        evaluation : [EVALUATION]"""
    from
        "openai/text-davinci-002"
    distribution
        EVALUATION in ["1", "0"]
    '''


async def is_correct(prediction: str, ground_truth: str):
    evaluation = await evaluate_(prediction, ground_truth)
    evaluation = evaluation.variables["EVALUATION"]
    if evaluation == "1":
        return True
    else:
        return False


if __name__ == "__main__":
    print(asyncio.run(is_correct("Hum berger", "humberger")))
