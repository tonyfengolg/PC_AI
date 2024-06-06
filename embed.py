from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

pdf_path = "docs/OLG HR Policy Combined.pdf"
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separator="\n"
)

docs = text_splitter.split_documents(documents=documents)

embeddings =  AzureOpenAIEmbeddings(model='text-embedding-ada-002',api_key=st.secrets['AZURE_OPENAI_API_KEY'],
    azure_endpoint=st.secrets['AZURE_OPENAI_ENDPOINT'])
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("docs/faiss_index_react")

