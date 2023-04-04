import openai
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import ZeroShotAgent, ConversationalAgent, Tool, AgentExecutor
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

from common.constants import *
from common.utils import to_text_list, file_to_df

openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# llm = OpenAI(model_name="text-davinci-003", temperature=0, max_tokens=700)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=700)
embeddings = OpenAIEmbeddings()
index_complaints = pinecone.Index("citizen-complaints-2023")
vectorstore_complaints = Pinecone(index_complaints, embeddings.embed_query, "complaint")
complaints_tool = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_complaints.as_retriever(search_kwargs={"k": 20}),
    verbose=True)

index_docs = pinecone.Index("urban-planning-documents-all")
vectorstore_docs = Pinecone(index_docs, embeddings.embed_query, "text")
docs_tool = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_docs.as_retriever(search_kwargs={"k": 3}),
    verbose=True)


# format_instructions = """Usa el siguiente formato:

# Question: [la pregunta que debes responder]
# Thought: [debes siempre pensar en que hacer]
# Action: [la accion que debes tomar, debe ser una de [{tool_names}]]
# Action Input: [la entrada para la accion]
# Observation: [el resultado de la accion]
# ... (este Thought/Action/Action Input/Observation se puede repetir N veces)
# Thought: Ahora sí se la respuesta final
# Final Answer: [la respuesta final explicada de forma detallada a la pregunta original]"""


# prompt = ZeroShotAgent.create_prompt(
#     tools,
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=["input", "agent_scratchpad"]
# )
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
tools = [
    Tool(
        name="Atención ciudadana",
        func=complaints_tool.run,
        description="Util para responder preguntas relacionadas con quejas, problematicas y/o propuestas ciudadanas. La entrada para esto debe ser un término de búsqueda único."
    ),
    Tool(
        name="Planeacion urbana",
        func=docs_tool.run,
        description="Util para responder preguntas relacionadas a planes, proyectos y diseños urbanos, contexto a problematicas y soluciones propuestas. La entrada para esto debe ser un término de búsqueda único."
    ),
]
tool_names = ", ".join([tool.name for tool in tools])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = ConversationalAgent.from_llm_and_tools(
    llm,
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    ai_prefix="Regina",
    human_prefix="Planeador Urbano",
    verbose=True)
# print(agent.template)
# agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
# agent = ConversationalAgent.from_llm_and_tools(
#     llm,
#     tools,
#     prefix=prefix,
#     suffix=suffix,
#     format_instructions=format_instructions, verbose=True)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
a = agent_executor.run(
    "Que se menciona en documentos urbanos respecto a la semaforizacion sincronizada y como esto esta relacionado con los problemas ciudadanos?")
print(a)
exit()
