from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from state import GlobalState
from researcher_agent import compiled_researcher
from analyst_agent import compiled_analyst

# --- 1. The Pydantic Router Schema ---
class RouterDecision(BaseModel):
    next_worker: Literal["Researcher", "Data Analyst", "FINISH"] = Field(
        ...,
        description="Select the next worker needed to fulfill the user's request, or FINISH if the final report can be generated from the current messages."
    )

model = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0.1)
# --- 2. The Supervisor Node ---
def supervisor_node(state: GlobalState) -> dict:
    system_prompt = """You are the Supervisor of an Intelligent Land Investment System.
    You manage two workers:
    - 'Researcher': Scrapes the web for live competitor data.
    - 'Data Analyst': Queries the internal SQL database for historical sales.
    
    Review the conversation history. Decide who needs to work next. If all necessary data is present in the chat history, select 'FINISH'."""
    
    # Bind the schema to force a structured output
    structured_llm = model.with_structured_output(RouterDecision)
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    decision = structured_llm.invoke(messages)
    
    # Update the state with the exact string ("Researcher", "Data Analyst", or "FINISH")
    return {"next_worker": decision.next_worker}

# --- 3. The Conditional Edge ---
def route_to_worker(state: GlobalState) -> str:
    # Read the decision made by the supervisor node
    return state["next_worker"]

# --- 4. Graph Construction ---
workflow = StateGraph(GlobalState)

# Add the Hub and the Spokes (The subgraphs plug in seamlessly as nodes)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Researcher", compiled_researcher)
workflow.add_node("Data Analyst", compiled_analyst)

workflow.set_entry_point("Supervisor")

# The Hub conditionally routes to the Spokes
workflow.add_conditional_edges(
    "Supervisor",
    route_to_worker,
    {
        "Researcher": "Researcher",
        "Data Analyst": "Data Analyst",
        "FINISH": END
    }
)

# The Spokes unconditionally route back to the Hub
workflow.add_edge("Researcher", "Supervisor")
workflow.add_edge("Data Analyst", "Supervisor")

app = workflow.compile()