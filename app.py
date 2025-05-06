import streamlit as st
import fitz  # from PyMuPDF
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
import uuid

os.environ["GOOGLE_API_KEY"] = st.secrets["my_secrets"]["GOOGLE_API_KEY"]
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Function to load and extract text from PDF
def load_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to fetch and extract text from URL
def load_url_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        # Remove scripts and styles
        for tag in soup(["script", "style", "noscript", "link", "meta"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return text.strip()
    except Exception as e:
        return f"Error fetching URL: {e}"

# Split text into chunks
def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

# Embed chunks and store in Chroma
def embed_and_store(chunks):
    persist_dir = f"chroma_store/{uuid.uuid4()}" 
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    return vectordb

# Query and get response
def query(question, vectordb):
    try:
        docs = vectordb.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25")
        prompt = f"""
        You are a helpful assistant. Based on the document and your own knowledge, answer the question below.
        Give a direct, accurate answer without referring to the document's contents or whether the answer is found there.

        Document:
        \"\"\"{context}\"\"\"

        Question: {question}
        Answer:"""
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"Error: {e}"

# Streamlit main app
def main():
    st.markdown("<h2 style='text-align: center;padding-bottom: 20px; font-size: 26px'>Document/URL Querying App</h2>", unsafe_allow_html=True)
    
    option = st.radio("Select Input Type:", ["Upload PDF", "Enter URL"])
    
    text = ""
    if option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf", label_visibility="collapsed")
        if uploaded_file:
            text = load_pdf(uploaded_file)
    elif option == "Enter URL":
        url = st.text_input("Enter a URL")
        if url:
            text = load_url_text(url)

    if text:
        chunks = split_text(text)
        vectordb = embed_and_store(chunks)
        question = st.text_input("Enter your query")
        if question:
            answer = query(question, vectordb)
            st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()