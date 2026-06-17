
import os
from pathlib import Path
from typing import Annotated, List, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader  # document_loader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# --- 1. GEMINI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()


class AgentState(TypedDict):
    question : str
    response : str
    document :List[Document] 

    
tools = []
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
    max_retries=3 # Built-in protection against connection drops
).bind_tools(tools)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def document_storage(relative_path: str):
    """
    Loads a PDF from a relative path string, splits it, 
    and indexes it into an InMemoryVectorStore.
    """
    # 1. Resolve the absolute path relative to where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.abspath(os.path.join(script_dir, relative_path))

    # 2. Safety check: Verify the file actually exists
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Could not find document at: {absolute_path}")

    try:
        # 3. Load the document directly from the file system
        loader = PyPDFLoader(absolute_path)
        docs = loader.load()

        # 4. Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        # 5. Initialize and populate the InMemoryVectorStore
        # This replaces FAISS completely and runs entirely in RAM
        vector_store = InMemoryVectorStore.from_documents(
            documents=all_splits, 
            embedding=embeddings
        )

        return vector_store

    except Exception as e:
        raise RuntimeError(f"Failed to process and index document: {e}")


document = document_storage("sample.pdf")  # Adjust the path as needed

def retrieve_relevant_docs(state: AgentState) -> AgentState:
    """
    Retrieves the top-k most relevant documents from the vector store based on the query.
    """
    try:
        results = document.similarity_search(state['question'])
        state['document'] = results
        return state
    except Exception as e:
        raise RuntimeError(f"Error during document retrieval: {e}")
    
def generate_response(state: AgentState) -> AgentState:
    """
    Generates a response based on the user's question and the retrieved documents.
    """
    try:
        relevant_docs = state['document']

        # Prepare the context for the LLM
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {state['question']}"

        # Generate the response using the LLM
        response = model.invoke([HumanMessage(content=prompt)])
        state['response'] = response.content

        return state
    except Exception as e:
        raise RuntimeError(f"Error during response generation: {e}")
    

graph = StateGraph(AgentState)

graph.add_node("retrieve_docs", retrieve_relevant_docs)
graph.add_node("generate_response", generate_response)

graph.set_entry_point("retrieve_docs")
graph.add_edge("retrieve_docs", "generate_response")
graph.add_edge("generate_response", END)

app = graph.compile()




if __name__ == "__main__":
    print("\n--- LangGraph RAG Agent Initialized ---")
    
    user_query = "What is the main topic of the document?"
    
    initial_state = {
        "question": user_query,
        "response": "",      
        "document": []       
    }
    
    print(f"Question: {user_query}")
    print("Running graph...\n")
    
    try:
        final_state = app.invoke(initial_state)
        
        print("--- Final Output ---")
        print(final_state["response"])
        
    except Exception as e:
        print(f"Graph execution failed: {e}")

