from typing import TypedDict , List
from langgraph.graph import StateGraph 

class AgentState(TypedDict):
    values: List[int]
    name: str
    results: str


def process_values(state: AgentState)-> AgentState:
    """This function handles multiple different inputs"""

    state["results"] = f"Hello {state['name']}! The sum of your values is {sum(state['values'])}."
    return state

graph = StateGraph(AgentState)

graph.add_node("processor", process_values)
graph.set_entry_point("processor")
graph.set_finish_point("processor")

app = graph.compile()

from IPython.display import Image,display

# --- THE NEW WAY TO GET THE IMAGE ---
png_bytes = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_bytes)
    
print("Graph saved to graph.png!")
# ------------------------------------

result = app.invoke({"name": "Alice", "values": [1, 2, 3, 4]})

print(result["results"])