import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os

# ============ Image Classifier Configuration ============
MODEL_PATH = 'model/best_model.h5'
model = load_model(MODEL_PATH)

class_mapping = {
    0: 'AbdomenCT',
    1: 'BreastMRI',
    2: 'Hand',
    3: 'CXR',
    4: 'HeadCT',
    5: 'ChestCT'
}

privacy_info = {
    'HeadCT': {"Privacy Level": "High", "Privacy Budget (ε)": "0.5", "Noise Level": "High noise"},
    'BreastMRI': {"Privacy Level": "High", "Privacy Budget (ε)": "0.5", "Noise Level": "High noise"},
    'AbdomenCT': {"Privacy Level": "Medium", "Privacy Budget (ε)": "1.0", "Noise Level": "Moderate noise"},
    'ChestCT': {"Privacy Level": "Medium", "Privacy Budget (ε)": "1.0", "Noise Level": "Moderate noise"},
    'CXR': {"Privacy Level": "Low", "Privacy Budget (ε)": "2.0", "Noise Level": "Minimal noise"},
    'Hand': {"Privacy Level": "Low", "Privacy Budget (ε)": "2.0", "Noise Level": "Minimal noise"}
}

# ============ RAG Chatbot Configuration ============
folder_path = "db"
os.makedirs(folder_path, exist_ok=True)

cached_llm = Ollama(model="wizardlm2:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)

# Load documents from knowledge folder
def setup_vectorstore():
    knowledge_path = "knowledge"
    docs = []
    for filename in os.listdir(knowledge_path):
        if filename.endswith(".pdf"):
            loader = PDFPlumberLoader(os.path.join(knowledge_path, filename))
            docs.extend(loader.load_and_split())
    chunks = text_splitter.split_documents(docs)
    vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=folder_path)
    vector_store.persist()
    return vector_store

if not os.path.exists(os.path.join(folder_path, 'chroma-collections.parquet')):
    setup_vectorstore()

raw_prompt = PromptTemplate.from_template(
    """<s>[INST] You are a medical assistant. Answer with reliable medical advice if known, otherwise, respond 'I'm not sure about this'. [/INST] </s>
    [INST] {input} Context: {context} Answer: [/INST]"""
)

def retrieve_answer(query):
    try:
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 20, "score_threshold": 0.6}
        )
        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)
        result = chain.invoke({"input": query})
        sources = [{"source": doc.metadata.get("source", ""), "content": doc.page_content} for doc in result["context"]]
        return result["answer"], sources
    except Exception as e:
        st.error(f"Chatbot error: {e}")
        return None, None

# ============ Streamlit UI ============
st.title("🧠 Medical Assistant: Image Classifier & Chatbot")

tab1, tab2 = st.tabs(["🩻 Image Classification", "💬 Medical Chatbot"])

# ============ Tab 1: Image Classifier ============
with tab1:
    st.header("Upload Medical Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((64, 64))
        image = np.array(image).astype('float32') / 255.0
        image = np.expand_dims(image, axis=(0, -1))

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        predictions = model.predict(image)
        predicted_label = np.argmax(predictions, axis=1)[0]
        predicted_class = class_mapping[predicted_label]

        st.subheader("Prediction:")
        st.write(f"**Predicted Class:** {predicted_class}")

        if predicted_class in privacy_info:
            info = privacy_info[predicted_class]
            st.subheader("Privacy Information:")
            st.write(f"**Privacy Level:** {info['Privacy Level']}")
            st.write(f"**Privacy Budget (ε):** {info['Privacy Budget (ε)']}")
            st.write(f"**Noise Level:** {info['Noise Level']}")
        else:
            st.warning("Privacy information not available for this class.")

# ============ Tab 2: Chatbot ============
with tab2:
    st.header("Ask a Medical Question")
    query = st.text_input("Enter your question:")

    if st.button("Submit", key="chat_submit") and query:
        answer, sources = retrieve_answer(query)
        if answer:
            st.subheader("Answer:")
            st.write(answer)

            if sources:
                with st.expander("Sources (from knowledge PDFs)"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {src['source']}")
                        st.write(src["content"])
        else:
            st.warning("No valid answer found. Try rephrasing the question.")
