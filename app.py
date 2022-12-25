from flask import Flask
from flask import request
from utils import *

app = Flask(__name__)


@app.get("/")
def hello_world():
    return "<p>Hello, World!</p>"

# post endpoint to submit query and generate embeddings
# takes in query string and returns string


@app.post("/context")
def get_context():
    question = request.form['question']
    context = create_embeddings(question)
    return {"context": context}


def create_embeddings(question):
    # do something with question
    df = ddg_extract(question)
    df = get_df_embeddings(df)
    add_to_pinecone(df)
    context = construct_prompt(question)
    return "context"
