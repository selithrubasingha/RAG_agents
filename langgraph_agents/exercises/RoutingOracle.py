

import os
from pathlib import Path
from typing import Annotated, List, Sequence, TypedDict , Literal
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
from pydantic import BaseModel, Field
from tavily import TavilyClient
# --- 1. GEMINI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# Define the exact schema we want the LLM to output
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal["vector_db", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web_search or a vector_db.",
    )

class AgentState(TypedDict):
    question : str
    response : str
    document :List[Document] 
    tool_decision : str
    
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_web(query: str, max_results: int = 3) -> str:
    """
    Searches the live internet using the Tavily API to find up-to-date information, 
    news, or factual data that is not present in your local database.
    Use this tool whenever the user asks about current events, recent news, 
    or topics requiring real-time web context.
    
    Args:
        query: The specific search query string to look up.
        max_results: The maximum number of search results to return (default: 3).
    """
    try:
        # Perform the search. Tavily returns a dictionary with a 'results' list
        response = tavily_client.search(query=query, max_results=max_results)
        
        # Extract the results safely
        results = response.get("results", [])
        
        if not results:
            return "No relevant web results found for that query."

        # Format the output cleanly so the LLM can read and cite it easily
        formatted_results = []
        for result in results:
            formatted_results.append(
                f"Title: {result['title']}\n"
                f"Source: {result['url']}\n"
                f"Content: {result['content']}"
            )
            
        # Join them together with a clear separator
        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        return f"Tool Execution Error: Web search failed with error: {str(e)}"
    
tools = [search_web]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
    max_retries=3 # Built-in protection against connection drops
)

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

def router_node(state: AgentState) -> AgentState:
    """
    An LLM node that evaluates the query and returns a routing decision.
    """
    # 1. Provide strict instructions to the LLM
    system_prompt = """You are an expert router for a Land Investment System.
    The vector database contains documents regarding general property zoning laws and regulations.
    Use the web_search tool for questions about current events, live interest rates, or recent news.
    Route the user's query to the appropriate datasource."""

    # 2. Bind the Pydantic schema to the model
    # Notice we use the same 'model' from your previous code
    structured_llm = model.with_structured_output(RouteQuery)
    
    # 3. Invoke the model
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"])
    ]
    
    # The result will be a Pydantic object, not a raw string!
    result = structured_llm.invoke(messages)
    
    # We update the state with the LLM's decision so the conditional edge can read it
    state["tool_decision"] = result.datasource
    return state

def web_search_node(state: AgentState) -> AgentState:
    """
    Executes the Tavily web search tool and wraps the string result 
    into a Document object so generate_response can read it.
    """
    print("--- RUNNING WEB SEARCH ---")
    query = state["question"]
    
    # Call the tool directly using .invoke()
    search_result_str = search_web.invoke({"query": query})
    
    # Wrap the string in a Document object to maintain compatibility with your state
    web_document = Document(
        page_content=search_result_str, 
        metadata={"source": "tavily_web_search"}
    )
    
    # Overwrite the document state with the web result
    state['document'] = [web_document]
    return state

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
    Generates a response based on the user's question and the retrieved documents or the web search results.
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

def decide_next_node(state: AgentState) -> str:
    """Reads the LLM's decision and routes to the correct node."""
    
    decision = state.get("tool_decision")
    
    if decision == "web_search":
        print("Routing -> Web Search")
        return "web_search_node"  # The name of the node where you call Tavily
    else:
        print("Routing -> Vector DB")
        return "retrieve_docs_node" # The name of your Phase 1 vector search node  

graph = StateGraph(AgentState)

# 1. Add all four nodes
graph.add_node("router", router_node)
graph.add_node("web_search_node", web_search_node)
graph.add_node("retrieve_docs_node", retrieve_relevant_docs)
graph.add_node("generate_response", generate_response)

# 2. Set the starting point
graph.set_entry_point("router")

# 3. Add the conditional edges branching out of the router
graph.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "web_search_node": "web_search_node",
        "retrieve_docs_node": "retrieve_docs_node",
    }
)

# 4. Connect BOTH retrieval nodes to the generator
graph.add_edge("web_search_node", "generate_response")
graph.add_edge("retrieve_docs_node", "generate_response")

# 5. Connect the generator to the END
graph.add_edge("generate_response", END)


app = graph.compile()




if __name__ == "__main__":
    print("\n--- LangGraph RAG Agent Initialized ---")
    
    user_query = "What are the interest rates in Colombo srilanka area Land market as of this week?"
    
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

