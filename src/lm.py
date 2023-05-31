import os
from abc import ABC, abstractmethod

import dotenv
import openai

dotenv.load_dotenv()

from pyllamacpp.model import Model
from transformers import pipeline, set_seed


class LM(ABC):
    @abstractmethod
    def __call__(self, prompt: str, **kwargs: any) -> str:
        pass


class GPT3(LM):
    def __init__(self) -> None:
        super().__init__()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def __call__(self, prompt: str) -> str:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.1,
            max_tokens=100,
            presence_penalty=1,
        )
        return response.choices[0].text


class GPT2(LM):
    def __init__(self) -> None:
        super().__init__()
        set_seed(42)
        self.generator = pipeline("text-generation", model="gpt2")

    def __call__(self, prompt: str, stop_sequence: str = "\n") -> str:
        output = self.generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            return_full_text=False,
            stop_sequence=stop_sequence,
        )
        return output[0]["generated_text"]


class LlamaQ4(LM):
    def __init__(self, model_path: str = "models/llama/7B/ggml-model-q4_0.bin") -> None:
        super().__init__()
        self.model_path = model_path

    def __call__(self, prompt: str, stop_sequence: str = "\n") -> str:
        self.model = Model(model_path=self.model_path)
        output = self.model.generate(prompt=prompt, antiprompt=stop_sequence)
        output = "".join(output)
        return output