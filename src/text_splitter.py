from abc import ABC, abstractmethod

from langchain.text_splitter import TokenTextSplitter


class Splitter(ABC):
    @abstractmethod
    def split(self, text: str) -> list["str"]:
        pass


class TokenSplitter(Splitter):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 0) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_text(text)


if __name__ == "__main__":
    output = TokenSplitter(chunk_size=2).split("Hello, my dog is cute")
    print(output)
