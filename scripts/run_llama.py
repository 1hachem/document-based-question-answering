from src.lm import Llama

prompts = ["hello llama", "how is the weather in dreamland ?"]
llm = Llama()
llm(prompts)
