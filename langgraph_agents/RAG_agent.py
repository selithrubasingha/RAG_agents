import os
from pathlib import Path
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# --- 1. GEMINI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# --- 2. LINUX PATH HANDLING ---
# Creates a 'drafts' directory in your current working directory (~/Code/ai_agents/.../drafts)
DRAFTS_DIR = Path.cwd() / "drafts"
DRAFTS_DIR.mkdir(exist_ok=True)

document_content = ""

# --- 3. VECTOR EMBEDDINGS SETUP ---
# Initialize the Gemini embedding model and an in-memory vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embeddings)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    
    # Automatically embed and store this draft version for semantic search
    vector_store.add_texts(
        texts=[content], 
        metadatas=[{"status": "draft_update"}]
    )
    
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def search_previous_drafts(query: str) -> str:
    """Searches previous document updates for specific concepts or phrases using vector embeddings."""
    results = vector_store.similarity_search(query, k=2)
    if not results:
        return "No relevant past drafts found."
    
    found_texts = "\n\n---\n\n".join([doc.page_content for doc in results])
    return f"Found these relevant past drafts:\n{found_texts}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process."""
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
        
    # Use pathlib for safe, cross-platform path resolution on Linux
    file_path = DRAFTS_DIR / filename

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {file_path}")
        return f"Document has been saved successfully to '{file_path}'."
    except Exception as e:
        return f"Error saving document: {str(e)}"

# Register the new embeddings tool
tools = [update, save, search_previous_drafts]

# --- 4. INITIALIZE GEMINI LLM ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
    max_retries=3 # Built-in protection against connection drops
).bind_tools(tools)

def our_agent(state: AgentState) -> dict:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You help the user update and modify documents.
    
    - Use 'update' to change the document.
    - Use 'search_previous_drafts' to recall past document versions or specific ideas we removed.
    - Use 'save' to write the final text to disk.
    
    The current document content is:\n{document_content}
    """)

    all_messages = [system_prompt] + list(state["messages"])
    response = model.invoke(all_messages)
    
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    if not messages:
        return "agent"
        
    last_message = messages[-1]
    
    # If the save tool just ran successfully, terminate the graph
    if isinstance(last_message, ToolMessage) and "saved successfully" in last_message.content.lower():
        return "end"
        
    # Otherwise, loop back to the agent to read the tool output
    return "agent"

# --- GRAPH ARCHITECTURE ---
graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

# Use LangGraph's built-in tools_condition to route safely
graph.add_conditional_edges("agent", tools_condition)
graph.add_conditional_edges("tools", should_continue, {"agent": "agent", "end": END})

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    # The graph state is initialized cleanly
    state = {"messages": []}
    
    # The interactive loop runs OUTSIDE the graph
    while True:
        user_input = input("\n👤 USER: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
            
        state["messages"].append(HumanMessage(content=user_input))
        
        for step in app.stream(state, stream_mode="values"):
            message = step["messages"][-1]
            
            # Print AI text safely
            if isinstance(message, AIMessage) and message.content:
                print(f"\n🤖 AI: {message.content}")
                
            # Print Tool outputs
            if isinstance(message, ToolMessage):
                print(f"\n🛠️ TOOL RESULT: {message.content}")
                
        # Persist state for the next loop iteration
        state = step 

    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()