import os
import warnings

import pandas as pd
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from pymongo import MongoClient

from common.constants import *
from common.utils import file_to_df

warnings.filterwarnings('ignore')

FILE_PATH = 'data/pdfs/'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index = pinecone.Index(PINECONE_INDEX_NAME)


def save_to_db(df: pd.DataFrame, db_name: str):
  client = MongoClient(MONGODB_CONNECTION)
  db = client[MONGODB_PROJECT]
  collection = db[db_name]

  collection.delete_many({})
  data = df.to_dict(orient='records')
  data = [{'_id': doc.pop(C_REPORT_ID), **doc} for doc in data]
  result = collection.insert_many(data)
  return result


def generate_docs_vectorstore(data) -> Pinecone:
  index.delete(delete_all=True, namespace=data['namespace'])
  directory_loader = DirectoryLoader(FILE_PATH, glob='**/*SP.pdf', loader_cls=PyPDFLoader)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  docs = directory_loader.load_and_split(text_splitter)
  print('num splitted docs', len(docs))
  embeddings = OpenAIEmbeddings()
  db = Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME, namespace=data['namespace'])
  return db


def generate_complaints_vectorstore(data) -> Pinecone:
  index.delete(delete_all=True, namespace=data['namespace'])
  df = file_to_df(data['file'])
  df = df.set_index(C_REPORT_ID, drop=False)
  docs = []
  for i, row in df.iterrows():
    meta = df[KEEP_COLUMNS].to_dict('records')
    doc = Document(page_content=row[data['embed_name']], metadata=row[KEEP_COLUMNS])
    docs.append(doc)
  ids = [x.metadata[C_REPORT_ID] for x in docs]
  print('num splitted docs', len(docs))
  embeddings = OpenAIEmbeddings()
  db = Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME,
                               namespace=data['namespace'], ids=ids)
  return db


MAPPINGS = [
    {
        "db": {"name": "complaints", "file": FILE_CITIZEN_REPORTS},
        "vectorstore": {
            "preprocess": generate_complaints_vectorstore,
            "file": FILE_CITIZEN_REPORTS,
            "namespace": "complaints",
            "embed_name": C_COMPLAINT
        }
    },
    # {
    #     "vectorstore": {
    #         "preprocess": generate_docs_vectorstore,
    #         "namespace": "docs"
    #     }
    # }
]


if __name__ == "__main__":
  for x in MAPPINGS:
    if x.get("db"):
      df = file_to_df(x["db"]["file"])
      save_to_db(df, x["db"]["name"])
    if x.get("vectorstore"):
      x["vectorstore"]["preprocess"](x["vectorstore"])
