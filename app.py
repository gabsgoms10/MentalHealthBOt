import streamlit as st # for Web App
from streamlit_chat import message # For chat
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import ctransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory