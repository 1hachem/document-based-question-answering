import os
from abc import ABC, abstractmethod

import dotenv
import openai

dotenv.load_dotenv()

import torch
from pyllamacpp.model import Model
from transformers import (AutoModelForCausalLM, AutoTokenizer, pipeline,
                          set_seed)


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
            max_tokens=20,
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
            max_new_tokens=20,
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
        output = self.model.generate(
            prompt=prompt, antiprompt=stop_sequence, n_predict=20
        )
        output = "".join(output)
        return output


class GPTJ(LM):
    def __init__(self) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, prompt: str, stop_sequence: str = "\n") -> str:
        pipeline_ = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            stop_sequence=stop_sequence,
            device=self.device,
        )
        output = pipeline_(prompt, max_new_tokens=20, return_full_text=False)
        return output[0]["generated_text"]


if __name__ == "__main__":
    llm = GPTJ()
    output = llm("Hello, my dog is cute")
    print(output)
