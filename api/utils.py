# import everything we need
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import ddg
import pandas as pd
from dotenv import load_dotenv
import os
import openai
import re
# annoying coz conda doesnt list it, so have to install pip in venv and then use the venv's pip to install it
import pinecone
from tqdm.auto import tqdm
import psycopg2

# set constants
EMBEDDINGS_MODEL = "text-embedding-ada-002"
EMBEDDINGS_DIMENSION = 1536
PINECONE_BATCH_SIZE = 32
MAX_SECTION_LEN = 500
SEPARATOR = "\n "
CLEANR = re.compile('<.*?>')

# load env variables
load_dotenv()

# setup openai and pinecone
openai.api_key = os.environ.get('OPENAI_API_KEY')
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment="us-west1-gcp"
)
if openai.api_key is None:
    print("openai api key not found")
if pinecone is None:
    print("pinecone api key not found")

index = pinecone.Index('openai')

# Connect to your postgres DB
conn = psycopg2.connect(os.environ.get('DB'))

# 1000 tokens ~ 750 words; there is no way to get the number of tokens from the API for 2nd gen models for now
# 1 token ~ 4 characters


def token_estimate(text):
    # anything above 8000 tokens is too long for the ada model
    return len(text) / 4

# we know that openai ada model costs $0.0004 / 1K tokens


def cost_estimate(tokens):
    return tokens / 1000 * 0.0004

# get embeddings for text


def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(
        model=EMBEDDINGS_MODEL,
        input=text
    )
    return result["data"][0]["embedding"]

# add embeddings to pinecone index


def add_to_pinecone(df: pd.DataFrame):
    for i in tqdm(range(0, df.shape[0], PINECONE_BATCH_SIZE)):
        # set end position of batch
        i_end = min(i+PINECONE_BATCH_SIZE, df.shape[0])
        # slice df
        temp_df = df.loc[i: i_end]
        # get batch of lines and IDs
        ids_batch = [str(n) for n in range(i, i_end)]
        # prep metadata and upsert batch
        meta = [{'content': line} for line in temp_df['content'].values]
        embeds = temp_df['embeddings'].values
        to_upsert = zip(ids_batch, embeds, meta)
        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert))


# calculate embeddings and enforce token rules for any df
# run this function once your parser has created a df with columns 'title', 'heading', 'content'
def get_df_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df['tokens'] = df['content'].apply(token_estimate)
    # filter tokens by 40-8000
    df = df[df['tokens'] > 40]
    df = df[df['tokens'] < 8000]
    # get embeddings
    df['embeddings'] = df['content'].apply(get_embedding)
    return df


def construct_prompt(question: str) -> str:
    """
    Fetch relevant context for a question, and construct a prompt
    """
    query_embedding = get_embedding(question)
    res = index.query([query_embedding], top_k=5, include_metadata=True)
    header = """Context:\n"""
    token_len = 2 + token_estimate(question)

    for match in res["matches"]:
        # compute token length for match metadata
        metadata = match["metadata"]["content"]
        metadata_len = token_estimate(metadata)
        # one for the separator
        token_len += metadata_len + 1
        if token_len > MAX_SECTION_LEN:
            # add as much as u can
            header += metadata[:MAX_SECTION_LEN -
                               int(token_len)] + SEPARATOR + "..."
            break
        header += metadata + SEPARATOR
    return header


def ddg_extract(question: str) -> pd.DataFrame:
    """
    Get top 3 links for a given query
    Get results from 2022
    extract all <p> tags and put them in pinecone
    if reseult already embedded, skip it
    """
    # get top 3 links
    results = ddg(question, region='wt-wt', safesearch='Off', max_results=2)
    df = pd.DataFrame(columns=['content'])
    # get text from each link
    for result in results:
        # check if link is already in db
        if check_if_in_db(result['href']):
            continue
        # add link to db
        add_to_db(result['href'])
        # get text from link
        soup = BeautifulSoup(requests.get(result['href']).text, 'html.parser')
        p_tags = soup.find_all('p')
        df1 = pd.DataFrame(columns=['content'])
        df1['content'] = [p_tag.text for p_tag in p_tags]
        df = pd.concat([df, df1], ignore_index=True)
    return df


def add_to_db(url):
    # Open a cursor to perform database operations
    cur = conn.cursor()
    # Execute a command: this creates a new table
    cur.execute("INSERT INTO websites_seen (url) VALUES (%s)", (url,))
    # Make the changes to the database persistent
    conn.commit()
    # Close communication with the database
    cur.close()


def check_if_in_db(url):
    # Open a cursor to perform database operations
    cur = conn.cursor()
    # Execute a command: this creates a new table
    cur.execute("SELECT * FROM websites_seen WHERE url = %s", (url,))
    # Make the changes to the database persistent
    res = cur.fetchone()
    # Close communication with the database
    cur.close()
    # return false if not in db
    return res is not None
