from src.lm import Llama

prompts = [
    """question : What is the main reason consulting pharmacists are increasingly working directly with patients?
    context : This trend may be gradually reversing as consultant pharmacists begin to work directly with patients, primarily because many elderly people are now taking numerous medications but continue to live outside of institutional settings.
    answer :""",
]
llm = Llama()
llm(prompts)
