import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, create_history_aware_retriever
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from typing import Any, List
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool, tool, BaseTool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import PromptTemplate
from typing import Any
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub

load_dotenv()

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
INDEX_NAME = os.environ.get('INDEX_NAME')

def run_modification_agent(query: str, chat_history: List[tuple[str, Any]] = []):
    base_prompt = hub.pull("hwchase17/react-chat")
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0.2, model="gpt-4-turbo"), chain_type='stuff',
                                     retriever=docsearch.as_retriever(),
                                     return_source_documents=True)

    def qa_executor_wrapper(originalprompt: str) -> dict[str, Any]:
        return qa({"query": originalprompt, "chat_history": chat_history})

    tools = [
        Tool(
            name='Resume Retrieval and Modification Agent',
            func=qa_executor_wrapper,
            description="Searches and returns key contexts from the User Resume that is stored in vector database and modifies the context based user's input and Returns the modified Resume"
        )
    ]

    memory = ChatMessageHistory(session_id="test-session")
    llm = ChatOpenAI(temperature=0, model='gpt-4-turbo')
    prompt = base_prompt.partial(instructions="")

    agent = create_react_agent(
        prompt=prompt,
        llm=llm,
        tools=tools
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": f"{query}", "chat_history": chat_history})
    modified_resume = result.get('output')

    return modified_resume

def run_python_agent(modified_resume: str, chat_history: List[tuple[str, Any]] = []):
    instructions = """
    You are an agent designed to write and execute python code to answer questions. 
    You have access to a python REPL, which you can use to execute python code. If you get an error, debug your code and try again. 
    Only use the output of your code to answer the question. You might know the answer without running any code, but you should still run the code to get the answer. 
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt_py = hub.pull("langchain-ai/react-agent-template")
    python_prompt = base_prompt_py.partial(instructions=instructions)
    python_tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=python_prompt,
        llm=ChatOpenAI(temperature=0.2, model="gpt-4-turbo"),
        tools=python_tools
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=python_tools, verbose=True)

    def python_executor_wrapper(originalprompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": originalprompt, "chat_history": chat_history})

    save_query = f"""
                Write a python code to convert the following resume text to a DOCX file to be saved in the current working directory. Name the file as the name of the candidate and the role they are modifying it for. Also show the current working directory Make sure to us os.path to save the file in right path . Key modifications to be addressed during the saving of the text is that  the name of the candidate should be in bigger font, bold, and centered at the top.
                Make any sub-headings bold and ensure appropriate spacing and aesthetics for a professional look.
                The final document should fit on one page. Use the tools you have at your disposal.
                Resume text: {modified_resume}"""

    result = python_executor_wrapper(save_query)

    return result.get('output')
