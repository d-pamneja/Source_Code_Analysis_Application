# Importing the dependencies
from src.Source_Code_Analysis.utils import ingest_repo,load_repo,chunk_split,load_embeddings
from langchain.vectorstores import Chroma
from src.Source_Code_Analysis.logger import logging

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PreRequisties
DATA_DIRECTORY = "data/"

# Function to initialize or reload the database

# PreRequisites
documents = load_repo("./repo/")
text_chunks = chunk_split(documents)
embeddings = load_embeddings()

# Create or reload Vector Database
vectordb = Chroma.from_documents(
    text_chunks,
    embedding=embeddings,
    persist_directory=DATA_DIRECTORY
)
vectordb.persist()

logging.info(f"Vector DB Created or reloaded.")

