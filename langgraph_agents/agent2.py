
from typing import TypedDict  , List , Union
from langchain_core.messages import HumanMessage , AIMessage
from langgraph.graph import StateGraph, START , END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # New Import

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7) 

def process(state: AgentState)->AgentState:
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"Agent Response: {response.content}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history= []

user_input = input("Enter: ")

while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")
