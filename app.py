import streamlit as st
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["my_secrets"]["GOOGLE_API_KEY"]

# Function to load PDF from uploaded file
def load_doc(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to split documents into chunks
def split_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

# Function to embed chunks and store in Chroma vector DB
def embed_and_store(chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory="chroma_store"
    )
    return vectordb

# Function to retrieve top relevant chunks using similarity search and generate an answer
def query(question, vectordb):
    try:
        docs = vectordb.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        prompt = f"""
        You are a helpful assistant. Based on the document and your own knowledge, answer the question below.
        Give a direct, accurate answer without referring to the document's contents or whether the answer is found there." 

        Document:
        \"\"\"{context}\"\"\"

        Question: {question}
        Answer:"""
        return llm.invoke(prompt).content.strip()

    except Exception as e:
        return f"Error: {e}"

def main():
    st.markdown("""<h2 style='text-align: center;padding-bottom: 20px; font-size: 26px'>Document Querying Application</h2>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf", label_visibility="collapsed")
    
    if uploaded_file:
        text = load_doc(uploaded_file)
        chunks = split_text(text)
        vectordb = embed_and_store(chunks)
        question = st.text_input("Enter your query")
        if question:
            answer = query(question, vectordb)
            st.write(f"Answer: {answer}")
            
if __name__ == "__main__":
    main()
