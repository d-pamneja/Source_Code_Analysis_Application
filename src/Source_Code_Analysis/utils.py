# Importing the dependencies

import os
import shutil
from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from src.Source_Code_Analysis.logger import logging

# Repository Ingestion
def ingest_repo(repo_url):
    os.makedirs("repo",exist_ok=True)
    repo_path = "./repo/"
    
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
        
        
    Repo.clone_from(repo_url, to_path=repo_path)
    
# Loading the Repository
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py",".ipynb"], # Will load only python files, remove if you want to load multiple file types
        parser= LanguageParser(language=Language.PYTHON,parser_threshold=500)
    )
    
    documents = loader.load()
    
    logging.info("Repository Loaded")
    return documents

# Chunking the documents
def chunk_split(documents):
    document_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size = 2000,
        chunk_overlap = 200,
        )
    
    text_chunks = document_splitter.split_documents(documents=documents)
    logging.info(f"Code converted to chunks of size : {len(text_chunks)}")
    return text_chunks

# Load Embeddings
def load_embeddings():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings


    
    
    
    

    