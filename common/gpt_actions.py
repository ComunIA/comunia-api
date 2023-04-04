import os
from time import time, sleep

import openai
from openai import Embedding, ChatCompletion, Completion
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate
from langchain.docstore.document import Document
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.agents import ConversationalAgent, AgentExecutor
from langchain import LLMChain
from common.constants import *
from common.utils import to_text_list, file_to_df, chunk_list
from langchain.chains import ConversationalRetrievalChain
from typing import Dict, List, Tuple, Optional, Any
import pinecone


openai.api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=700)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index = pinecone.Index('ai-urban-planning')
embeddings = OpenAIEmbeddings()
complaints_db = Pinecone(index, embeddings.embed_query, 'complaint')
# docs_db = Pinecone(persist_directory=DOCS_INDEX_DIR, embedding_function=embeddings)


_template = """Dada la siguiente conversacion y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente.
Historial del chat:
{chat_history}
Siguiente pregunta: {question}
Pregunta independiente:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Utiliza los siguientes fragmentos de contexto para responder a la pregunta al final. Si no conoces la respuesta, da información que podria estar relacionada o responde que no sabes, pero no intentes inventes una respuesta.
{context}
Pregunta: {question}
Respuesta util:"""
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# prompt_template2 = """Edita la respuesta original para que dar un contexto usando los fragmentos de documentos. Si no hay puntos utiles o relevantes escribe la respuesta original.
# FRAGMENTOS DE DOCUMENTOS:
# {context}

# PREGUNTA: {question}
# RESPUESTA ORIGINAL:
# {answer_complaints}

# RESPUESTA FINAL:"""
# QA_PROMPT2 = PromptTemplate(template=prompt_template2, input_variables=["context", "question", "answer_complaints"])


def question_complaints(question: str, chat_history: List[Tuple[str, str]],
                        vector_filter: Optional[Dict[str, Any]] = None) -> str:
  kwargs = dict(k=20, filter=vector_filter, namespace='complaints')
  questioning_complaints = ConversationalRetrievalChain.from_llm(
      chat,
      complaints_db.as_retriever(search_kwargs=kwargs),
      condense_question_prompt=CONDENSE_QUESTION_PROMPT,
      qa_prompt=QA_PROMPT,
      chain_type='stuff',
      verbose=True
  )
  answer_complaints = questioning_complaints(dict(question=question, chat_history=chat_history))['answer']
  # questioning_docs = ConversationalRetrievalChain.from_llm(
  #     chat,
  #     docs_db.as_retriever(search_kwargs={"k": 1}),
  #     condense_question_prompt=CONDENSE_QUESTION_PROMPT,
  #     qa_prompt=QA_PROMPT2,
  #     chain_type="stuff",
  #     verbose=True
  # )
  # question = f"{question}\nRespuesta en quejas ciudadanas: {answer_complaints}"
  # answer_docs = questioning_docs({"question": question, "chat_history": chat_history,
  #                                "answer_complaints": answer_complaints})['answer']
  return answer_complaints


def summarize_complaints(complaints: List[Any]) -> str:
  chunked_complaints = chunk_list(sorted(complaints), 1500)
  docs = [Document(page_content=to_text_list(x)) for x in chunked_complaints]

  initial_template = """Resume las preocupaciones clave, patrones y tendencias de las quejas ciudadanas para destacar los problemas más urgentes para los planificadores urbanos, manteniendo al mismo tiempo los matices de las experiencias individuales.
  QUEJAS CIUDADANAS:
  {text}

  RESUMEN:
  """
  initial_prompt = PromptTemplate(template=initial_template, input_variables=["text"])
  refine_template = """
  Tu trabajo es producir un resumen final.
  ------------
  RESUMEN ORIGINAL:
  {existing_answer}
  ------------
  QUEJAS CIUDADANAS:
  {text}
  ------------
  Dado las quejas ciudadanas, perfecciona el resumen original incluyendo preocupaciones clave, patrones y tendencias destacando los problemas más urgentes, manteniendo los matices de las experiencias individuales.
  El resumen final debe ser fluido, coherente y evitar repetir informacion.
  Si el contexto no es útil, devuelve el resumen original.

  RESUMEN FINAL:
  """
  # Edita el parrafo del reporte comunitario para incluir informacion
  # generalizada sobre las quejas ciudadanas que no se hayan mencionado
  # anteriormente.
  refine_prompt = PromptTemplate(
      input_variables=['existing_answer', 'text'],
      template=refine_template,
  )

  chain = load_summarize_chain(
      chat,
      chain_type='refine',
      question_prompt=initial_prompt,
      refine_prompt=refine_prompt,
      return_intermediate_steps=True
  )
  results = chain({'input_documents': docs}, return_only_outputs=True)
  return results['output_text']
