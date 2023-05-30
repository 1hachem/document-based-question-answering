import os
from abc import ABC, abstractmethod

import dotenv
import openai

dotenv.load_dotenv()

from transformers import pipeline, set_seed


class LM(ABC):
    @abstractmethod
    def __call__(self, prompt: str, **kwargs:any) -> str:
        pass

class GPT3(LM):
    def __init__(self) -> None:
        super().__init__()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def __call__(self, prompt: str) -> str:
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.9,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"],
        )
        return response.choices[0].text
    
class GPT2(LM):
    def __init__(self) -> None:
        super().__init__()
        set_seed(42)
        self.generator = pipeline('text-generation', model='gpt2')

    def __call__(self, prompt: str) -> str:
        output = self.generator(prompt, max_length=100, num_return_sequences=1)
        return output[0]['generated_text']


if __name__ == "__main__":
    prompt = "Once upon a time"
    # print(GPT3()(prompt))
    print("gpt2", GPT2()(prompt))