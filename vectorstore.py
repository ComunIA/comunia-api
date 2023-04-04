from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from common.constants import *
from common.utils import file_to_df

filePath = 'data/pdfs/'
index_name = 'ai-urban-planning'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index = pinecone.Index('ai-urban-planning')


def generate_docs_vectorstore(namespace: str) -> Pinecone:
  index.delete(delete_all=True, namespace=namespace)
  directory_loader = DirectoryLoader(filePath, glob='**/*SP.pdf', loader_cls=PyPDFLoader)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  docs = directory_loader.load_and_split(text_splitter)
  print('num splitted docs', len(docs))
  embeddings = OpenAIEmbeddings()
  df = Pinecone.from_documents(docs, embeddings, index_name=index_name, namespace=namespace)
  return df


def generate_complaints_vectorstore(file: str, namespace: str) -> Pinecone:
  index.delete(delete_all=True, namespace=namespace)
  df = file_to_df(file)
  df = df.set_index('report_id', drop=False)
  docs = []
  for i, row in df.iterrows():
    meta = df[KEEP_COLUMNS].to_dict('records')
    doc = Document(page_content=row['complaint'], metadata=row[KEEP_COLUMNS])
    docs.append(doc)
  ids = [x.metadata['report_id'] for x in docs]
  print('num splitted docs', len(docs))
  embeddings = OpenAIEmbeddings()
  db = Pinecone.from_documents(docs, embeddings, index_name=index_name, namespace=namespace)
  return db


if __name__ == '__main__':
  # generate_complaints_vectorstore(FILE_CITIZEN_REPORTS, 'complaints')
  generate_docs_vectorstore('docs')
