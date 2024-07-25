import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration of API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function for extracting text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function for creating chunks from extracted text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function for creating vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

template = """
You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer. If you do not get the context, please do not give the wrong answer to the question.

context:
{context}

question:
{user_question}
Answer:
"""

# Conversation chain
def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(input_variables=["context", "user_question"], template=template)
    chains = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chains

def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    # Ensure correct input keys
    context = " ".join([doc.page_content for doc in docs])
    response = chain({"input_documents": docs, "context": context, "user_question": user_question})

    st.write("**Answer : **", response['output_text'])

# Set Streamlit page configuration
st.set_page_config(
    page_title="PDF Chat",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="auto"
)

with st.sidebar:
    st.title("PDF Upload")
    pdf_docs = st.file_uploader("Upload the file", accept_multiple_files=True, type=["pdf"])

    if st.button("Submit"):
        with st.spinner("Uploading..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.success("VectorDB Upload Finished!")

def main():
    st.header("Chat with PDF")
    user_question = st.text_input("Ask Your Question")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
