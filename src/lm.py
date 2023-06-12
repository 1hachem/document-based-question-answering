from abc import ABC, abstractmethod

import dotenv

dotenv.load_dotenv()

import gc

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)


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


class Llama(LM):
    def __init__(self, model_path: str = "models/openllama/7B") -> None:
        super().__init__()
        self.model_path = model_path
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def __call__(self, requests: list[str]) -> list[str]:
        outputs = []
        for prompt in requests:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            generation_output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                temperature=0.0,
                num_return_sequences=1,
            )
            output = self.tokenizer.decode(
                generation_output[0], skip_special_tokens=True
            )
            output = output.strip(prompt)  # eq : return_full_sequence=False
            outputs.append(output)
            print(output)
        return outputs


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
        gc.collect()
        torch.cuda.empty_cache()
        self.model_name = "models/falcon/7B/snapshots/falcon"

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
            max_new_tokens=20,
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
    llm = Llama()
    output = llm(["Hello, my dog is cute", "hi my name is"])
    print(output)
