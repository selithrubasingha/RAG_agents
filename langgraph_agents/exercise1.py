from typing import Dict , TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict): # our state schema 
    message : str
    name : str


def complement_node(state : AgentState) -> AgentState:
    """Simple node that generates a greeting message."""

    state['message'] = state['name']+ ", You're doing an amazing job learning LangGraph !"

    return state

graph = StateGraph(AgentState)

graph.add_node("complementor", complement_node)

graph.set_entry_point("complementor")
graph.set_finish_point("complementor")

app = graph.compile()

from IPython.display import Image,display

# --- THE NEW WAY TO GET THE IMAGE ---
png_bytes = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_bytes)
    
print("Graph saved to graph.png!")
# ------------------------------------
result = app.invoke({"name": "Bob"})

print(result["message"])
