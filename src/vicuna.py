import torch
from fastchat.serve.inference import ChatIO, chat_loop

from src.lm import LM


class SimpleChatIO(ChatIO):
    def __init__(self) -> None:
        super().__init__()

    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

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
        self.chatio = SimpleChatIO()

    @torch.inference_mode()
    def __call__(
        self,
    ) -> str:
        chat_loop(
            self.model_path,
            self.device,
            num_gpus=self.num_gpus,
            load_8bit=self.load_8bit,
            temperature=1.0,
            repetition_penalty=1.0,
            max_new_tokens=10,
            debug=False,
            chatio=self.chatio,
            cpu_offloading=self.cpu_offloading,
            conv_template=None,
            max_gpu_memory=None,
        )


if __name__ == "__main__":
    llm = Vicuna()
    llm()
