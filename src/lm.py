import os
from abc import ABC, abstractmethod

import dotenv
import openai

dotenv.load_dotenv()

import torch
from pyllamacpp.model import Model

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed


class LM(ABC):
    @abstractmethod
    def __call__(self, requests: list[str], **kwargs: any) -> list[str]:
        pass


class GPT2(LM):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "gpt2"

    def __call__(self, requests: list[str], stop_sequence: str = "\n") -> list[str]:
        self.generator = pipeline("text-generation", model=self.model_name)
        sequences = self.generator(
            requests,
            max_new_tokens=20,
            num_return_sequences=1,
            return_full_text=False,
            stop_sequence=stop_sequence,
        )
        outputs = [seq[0]["generated_text"] for seq in sequences]
        return outputs


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

    def __call__(self, requests: list[str], stop_sequence: str = "\n") -> list[str]:
        pipeline_ = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            stop_sequence=stop_sequence,
            device=self.device,
        )
        sequences = pipeline_(
            requests, max_new_tokens=10, num_return_sequences=1, return_full_text=False
        )
        outputs = [seq[0]["generated_text"] for seq in sequences]
        return outputs


class Falcon(LM):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "tiiuae/falcon-7b"

    def __call__(self, requests: list[str]) -> list[str]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        pipeline_ = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        sequences = pipeline_(
            requests,
            max_new_tokens=10,
            do_sample=True,
            top_k=10,
            temperature=3e-4,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        outputs = [seq[0]["generated_text"] for seq in sequences]
        return outputs


if __name__ == "__main__":
    llm = GPTJ()
    output = llm(["Hello, my dog is cute", "hi my name is"])
    print(output)
