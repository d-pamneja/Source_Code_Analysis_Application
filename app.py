# Importing the dependencies
import os
from dotenv import load_dotenv

from src.Source_Code_Analysis.utils import ingest_repo,load_embeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

from flask import Flask,render_template,jsonify,request

# Loading the dependencies
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
embeddings = load_embeddings()
persist_directory = "data"

vectordb = Chroma( # Loading directly from the disk
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Creating the conversation chain
llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)
memory = ConversationSummaryMemory(
    llm = llm,
    memory_key="chat_history",
    return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectordb.as_retriever(
        search_type = "mmr", 
        search_kwargs = {"k":3}
    ),
    memory = memory
)


# Launching the application and routes
@app.route("/",methods = ["GET","POST"])
def index():
    return render_template("index.html")

@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        ingest_repo(user_input)
        os.system("python ./src/Source_Code_Analysis/indexing.py")

    return jsonify({"response": str(user_input) })

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    return str(result["answer"])


if __name__== '__main__':
    app.run(
        host = '0.0.0.0',
        port = 8080,
        debug=True
    )
    



