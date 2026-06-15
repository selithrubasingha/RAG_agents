from typing import Dict , TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict): # our state schema 
    message : str


def greeting_node(state : AgentState) -> AgentState:
    """Simple node that generates a greeting message."""

    state['message'] = "hey ! " +state['message']+ " how are you ?"

    return state

graph = StateGraph(AgentState)

graph.add_node("greeter", greeting_node)

graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

app = graph.compile()

from IPython.display import Image,display

# --- THE NEW WAY TO GET THE IMAGE ---
png_bytes = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_bytes)
    
print("Graph saved to graph.png!")
# ------------------------------------
result = app.invoke({"message": "Bob"})

print(result["message"])
