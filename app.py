from src.utils.utils import read_pdf
import streamlit as st
from src.lm import Llama, Falcon, GPT3
from src.vicuna import Vicuna
from src.embedding import Bert, E5, MiniLM
from src.models.simple_index import SimpleIndex


language_models = {
    "GPT-3": GPT3,
    "OpenLlama": Llama,
    "Vicuna": Vicuna,
    "Falcon": Falcon,
}
embedding_models = {"MiniLm": MiniLM, "E5": E5, "Bert": Bert}


# Streamlit app
def main():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            p {font-size:24px !important;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Left sidebar
    st.sidebar.title("Document-based question answering")
    desc = st.sidebar.markdown("End of studies project demo")
    name = st.sidebar.text("Hachem Betrouni ✨")

    # File upload
    uploaded_file = st.file_uploader("Upload a document 📄", type=["pdf"])
    if uploaded_file is not None:
        try:
            document = read_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Language model selection
    language_model = st.selectbox(
        "Select a Language Model 🤖",
        language_models.keys(),
        disabled=False if uploaded_file else True,
    )

    # Embedding model selection
    embedding_model = st.selectbox(
        "Select an Embedding Model 🔧",
        embedding_models.keys(),
        disabled=False if uploaded_file else True,
    )

    query = st.text_input(
        "Put your query here 👇",
        disabled=False if uploaded_file else True,
    )

    # Generate text and embeddings
    if st.button("Generate Answer"):
        index = SimpleIndex(
            embedding=embedding_models[embedding_model],
            llm=language_models[language_model],
        )
        answer = index(context=document, question=query)
        st.write("## Response ✨")
        st.write("#### " + answer)


if __name__ == "__main__":
    main()
