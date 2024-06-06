from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

embeddings =  AzureOpenAIEmbeddings(model='text-embedding-ada-002',api_key=st.secrets['AZURE_OPENAI_API_KEY'],
    azure_endpoint=st.secrets['AZURE_OPENAI_ENDPOINT'])
new_vectorstore = FAISS.load_local("docs/faiss_index_react", embeddings, allow_dangerous_deserialization=True)

def get_conversation_chain(vectorstore):
    llm = AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_deployment="gpt-4-32k",
    api_key=st.secrets['AZURE_OPENAI_API_KEY'],
    azure_endpoint=st.secrets['AZURE_OPENAI_ENDPOINT']
    )


    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.session_state.conversation = get_conversation_chain(
                    new_vectorstore)
    st.set_page_config(page_title="OLG P&C AI",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("OLG P&C AI:books:")
    user_question = st.chat_input("Ask a question about the policy:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()