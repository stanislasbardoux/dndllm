__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Langchain dependencies
import ollama
import logging
import langchain
from langchain_ollama import ChatOllama
import chainlit as cl
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
# from langchain_community.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
from langchain.prompts import ChatPromptTemplate # Import ChatPromptTemplate
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
from langchain.chains import RetrievalQA

langchain.debug = True;

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':True}

embedding_function = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

modelName="deepseek-r1:14b"
model = ChatOllama(model=modelName)
CHROMA_PATH = "chroma"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
chain = RetrievalQA.from_chain_type(model, retriever=db.as_retriever())

async def query_rag(query_text):
  try:
    return chain.invoke({"query": query_text})
  except Exception as e:
    error_message = f"An error occurred: {str(e)}"
    logging.error(f"Error: {error_message}")
    await cl.Message(content=error_message).send()

@cl.on_chat_start
async def start():
    logging.info("Chat started")
    cl.user_session.set("model", model)

@cl.on_message
async def main(message: cl.Message):
    logging.info(f"Received message: {message.content}")
    try:
        formatted_response, response_text = await query_rag(message.content)
        logging.info(f"Response: {response_text}")
        await cl.Message(content=response_text.content).send()
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(f"Error: {error_message}")
        await cl.Message(content=error_message).send()













#//////////// OLD METHODS //////////////

# Run : chainlit run main.py
DATA_PATH = "./data/"
def load_documents():
  document_loader = PyPDFDirectoryLoader(DATA_PATH)
  return document_loader.load() 

def split_text(documents: list[Document]):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True,
  )

  chunks = text_splitter.split_documents(documents)
  print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

  document = chunks[0]
  print(document.page_content)
  print(document.metadata)

  return chunks

def save_to_chroma(chunks: list[Document]):
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  db = Chroma.from_documents(
    chunks,
    embedding_function,
    persist_directory=CHROMA_PATH
  )
  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
  documents = load_documents()
  chunks = split_text(documents)
  save_to_chroma(chunks) 
# other chroma import : from langchain_chroma import Chroma # Importing Chroma vector store from Langchain