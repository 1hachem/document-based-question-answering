from src.utils.utils import read_pdf
import streamlit as st
from src.lm import Llama, Falcon, GPT3
from src.vicuna import Vicuna
from src.embedding import Bert, E5, MiniLM
from src.models.simple_index import SimpleIndex


language_models = {
    "OpenLlama": Llama,
    "Vicuna": Vicuna,
    "Falcon": Falcon,
    "GPT-3": GPT3,
}
embedding_models = {"E5": E5, "MiniLm": MiniLM, "Bert": Bert}


# Streamlit app
def main():
    st.title("Document-based question answering")

    # Left sidebar
    st.sidebar.title("Abstract")
    abstract = st.sidebar.markdown(
        """In this thesis, we present a comprehensive analysis of various open-source embedding models and large language models (LLMs) for document-based question answering (DBQA). 
        The goal of DBQA is to extract relevant information from a given document and answer user queries in natural language. 
        We evaluate the performance of different embedding models, including BERT, E5, and MiniLM, in combination with open-source LLMs such as Vicuna, Falcon, and OpenLlama.
	Our experimental setup involves a retriever-generator framework, where a retrieval system retrieves the most relevant contexts from a document using embeddings of document and query. 
    Then we condition the generative system (LLM) with the most promising context to generate a response to the user."""
    )
    name = st.sidebar.text("Hachem Betrouni")

    # File upload
    uploaded_file = st.file_uploader("Upload a document", type=["pdf"])
    if uploaded_file is not None:
        try:
            document = read_pdf(uploaded_file)
            st.text_area("Uploaded File", value=document, height=200)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Language model selection
    language_model = st.selectbox(
        "Select a Language Model",
        language_models.keys(),
        disabled=False if uploaded_file else True,
    )

    # Embedding model selection
    embedding_model = st.selectbox(
        "Select an Embedding Model",
        embedding_models.keys(),
        disabled=False if uploaded_file else True,
    )

    query = st.text_input(
        "Put your query here",
        disabled=False if uploaded_file else True,
    )

    # Generate text and embeddings
    if st.button("Generate Answer"):
        index = SimpleIndex(
            embedding=embedding_models[embedding_model],
            llm=language_models[language_model],
        )
        answer = index(context=document, question=query)
        st.write(answer)


if __name__ == "__main__":
    main()
