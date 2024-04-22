# Importing the dependencies
from src.Source_Code_Analysis.utils import ingest_repo,load_repo,chunk_split,load_embeddings
from langchain.vectorstores import Chroma
from src.Source_Code_Analysis.logger import logging

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PreRequisties
repo_path = "../../repo/"
print(os.system("pwd"))
documents = load_repo("./repo/")
text_chunks = chunk_split(documents)
embeddings = load_embeddings()

# Creating Vector DataBase
vectordb = Chroma.from_documents(
    text_chunks,
    embedding=embeddings,
    persist_directory= "data/"
)
vectordb.persist()

logging.info(f"Vector DB Created.")