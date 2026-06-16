from typing import TypedDict  , Annotated , Sequence 
from langchain_core.messages import BaseMessage  , SystemMessage , ToolMessage
from langgraph.graph import StateGraph, START , END
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # New Import
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Annotated - provides additional context without affecting the type itself
# Sequence - To automatically handle the state upadtes for sequences such as by adding new messages to a chat history

#reducer funtion 

# we can't alwyas append to the conversation history , it will get far too coplicated . 
# we can use a reducer function concatenate it to a string . 

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """This is an addition function that adds two numebrs together"""

    return a + b




@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

@tool
def divide(a: int, b: int):
    """Division function"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

tools = [add, subtract, multiply, divide]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

model = llm.bind_tools(tools)

def model_call(state: AgentState)->AgentState:
    
    system_prompt  = SystemMessage(
        content = "YOu are my AI assistant , please answer my query to the best of your ability."
        )
    response = model.invoke([system_prompt]+ state["messages"])
  
    return {"messages": [response]}

def should_continue(state: AgentState)->bool:
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node",tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges("our_agent", 
                            should_continue,
                            {
                                "continue":"tool_node",
                                "end": END,
                            },
                            )


graph.add_edge("tool_node","our_agent")

app = graph.compile()
    
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def print_stream2(stream):
    for s in stream:
        message = s["messages"][-1]
        
        if isinstance(message, tuple):
            print(message)
        # If the AI is giving a final answer (no tool calls)
        elif message.type == "ai" and not getattr(message, "tool_calls", None):
            print("\n================= Final Answer =================")
            # Check if Gemini returned the ugly list-of-dicts format
            if isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, dict) and 'text' in block:
                        print(block['text'])
            # Otherwise, just print the string normally
            else:
                print(message.content)
            print("================================================")
        # Otherwise, let it print the tool calls normally
        else:
            message.pretty_print()
inputs = {"messages": [("user", "Add 3+4 . then get the answer for it and multiply by 10. there after get the naswer for that and devide that by 7 . finally display the final answer ... if the final answer is 7 .. print'Hell Yea motha fuckers' if it is not 7 'Not so hell yea ! '. Also what is the capital of srilanka?")]}
print_stream2(app.stream(inputs, stream_mode="values"))















