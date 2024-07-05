from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from unstructured.partition.pdf import partition_pdf
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import base64
import os
import uuid
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma

pdf_path='docs/All Polices_Combined.pdf'


loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=4000, chunk_overlap=3000, separator="\n"
)

docs = text_splitter.split_documents(documents=documents)

embeddings =  AzureOpenAIEmbeddings(model='text-embedding-ada-002',api_key=st.secrets['AZURE_OPENAI_API_KEY'],
    azure_endpoint=st.secrets['AZURE_OPENAI_ENDPOINT'])
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("docs/faiss_index_react_all")

