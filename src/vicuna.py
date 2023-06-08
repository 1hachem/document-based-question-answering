from typing import Optional

import torch
from fastchat.model.chatglm_model import chatglm_generate_stream
from fastchat.model.model_adapter import get_conversation_template, load_model
from fastchat.serve.inference import ChatIO, generate_stream

from src.lm import LM


class SimpleChatIO:
    """this is a workaround to use fastchat for batched inference"""

    def __init__(self, requests: list[str]) -> None:
        self.requests = requests

    def prompt_for_input(self) -> str:
        return self.requests.pop(0) if self.requests else ""

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: SimpleChatIO,
    debug: bool,
):
    # Model
    model, tokenizer = load_model(
        model_path, device, num_gpus, max_gpu_memory, load_8bit, cpu_offloading, debug
    )
    is_chatglm = "chatglm" in str(type(model)).lower()
    is_fastchat_t5 = "t5" in str(type(model)).lower()

    # Hardcode T5 repetition penalty to be 1.2
    if is_fastchat_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    while True:
        conv = get_conversation_template(model_path)  # reset conversation

        try:
            inp = chatio.prompt_for_input()
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            generate_stream_func = chatglm_generate_stream
            prompt = conv.messages[conv.offset :]
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        output_stream = generate_stream_func(model, tokenizer, gen_params, device)
        outputs = chatio.stream_output(output_stream)

        yield outputs

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


class Vicuna(LM):
    def __init__(self, model_path: str = "models/vicuna/7B") -> None:
        super().__init__()
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = 3
        self.max_gpu_memory = None
        self.load_8bit = False
        self.cpu_offloading = False
        self.max_new_tokens = 20

    def __call__(
        self,
        requests: list[str],
    ) -> list[str]:
        chatio = SimpleChatIO(requests=requests)
        outputs = [
            output
            for output in chat_loop(
                self.model_path,
                self.device,
                num_gpus=self.num_gpus,
                load_8bit=self.load_8bit,
                temperature=0.0,
                repetition_penalty=1.0,
                max_new_tokens=20,
                debug=False,
                chatio=chatio,
                cpu_offloading=self.cpu_offloading,
                conv_template=None,
                max_gpu_memory=None,
            )
        ]
        return outputs


if __name__ == "__main__":
    requests = ["my name is hachem", "what is my name ?"]
    llm = Vicuna()
    outputs = llm(requests)
    print(outputs)
