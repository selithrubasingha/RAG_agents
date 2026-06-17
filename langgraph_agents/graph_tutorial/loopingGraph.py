

import random
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int

def greeting_node(state: AgentState) -> AgentState:
    """This node generates a greeting message"""
    state['name'] = f"Hello {state['name']} !"
    return state

def random_node(state: AgentState) -> AgentState:
    """This node does something random"""
    state['counter'] += 1
    state["number"].append(random.randint(0, 10))
    return state

def should_continue(state: AgentState) -> str:
    """This node decides whether to continue the loop or not"""
    if state['counter'] < 5:
        print("Entering loop iteration", state['counter'])
        return "loop"
    else:
        return "exit"  # Fixed: Matches the dictionary key

graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)
graph.add_node("random", random_node)

graph.set_entry_point("greeting")
graph.add_edge("greeting", "random")

graph.add_conditional_edges(
    "random",
    should_continue,
    {
        "loop": "random",
        "exit": END
    }
)
# Fixed: Deleted the conflicting graph.add_edge("random", END)

app = graph.compile()

result = app.invoke({"name": "Alice", "number": [], "counter": 0})

print("Generated numbers:", result["number"])