from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class GlobalState(TypedDict):
    # The continuous chat log shared across the entire system
    messages: Annotated[list, add_messages]
    
    # The supervisor's routing decision
    next_worker: str