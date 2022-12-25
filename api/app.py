from flask import Flask
from flask import request
from api.utils import *
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.get("/")
@cross_origin()
def hello_world():
    return "<p>Hello, World!</p>"

# post endpoint to submit query and generate embeddings
# takes in query string and returns string


@app.post("/context")
@cross_origin()
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
    return context
