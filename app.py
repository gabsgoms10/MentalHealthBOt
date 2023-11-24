import streamlit as st # for Web App
from streamlit_chat import message # For chat
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import ctransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory


# loading PDF
loader = DirectoryLoader("data/", glob=".pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunks_size=500, chunks_overlap=50)
text_chunks = text_splitter.split_documents(document)

# create embeddings
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})

# vectorstore
vector_store = faiss.from_documents(text_chunks, embeddings)

# create llm
llm = ctransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={"max_new_tokens":128, "temperature":0.01})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vector_store(search_kwargs={"k:2"}), memory=memory)