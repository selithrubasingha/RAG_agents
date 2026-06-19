
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_core.messages import HumanMessage
from supervisor import app  # Assuming your compiled graph is named 'app' in supervisor.py

if __name__ == "__main__":
    print("\n🚀 Initializing Intelligent Land Investment System - Multi-Agent Backend...\n")
    
    # A complex query designed to trigger BOTH the Data Analyst and the Web Researcher
    user_query = "What is the average price per perch for our internal sales in Colombo, and how does that compare to the current market rates being advertised online for Colombo land?"
    
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "next_worker": ""
    }
    
    print(f"User Query: {user_query}\n")
    print("-" * 40)
    
    # Stream the graph execution so we can watch the routing happen live
    # Stream the graph execution so we can watch the routing happen live
    for event in app.stream(initial_state, {"recursion_limit": 15}):
        for node_name, node_state in event.items():
            print(f"✅ Node Executed: {node_name}")
            
            # 1. Get the messages list from this specific node's update
            node_messages = node_state.get("messages", [])
            
            # 2. Only try to print if the node actually updated the messages list
            if node_messages:
                latest_message = node_messages[-1]
                if hasattr(latest_message, "content") and latest_message.content:
                     print(f"   Output preview: {latest_message.content[:150]}...\n")
            else:
                # If it's the supervisor, print its routing decision instead!
                next_worker = node_state.get("next_worker", "None")
                print(f"   Routing decision -> Next Worker: {next_worker}\n")
    print("-" * 40)
    print("🎯 Execution Complete!")
    
    # Print the final synthesized report from the chat history
    # The last message in the list will be the final agent's conclusion
    final_state = app.get_state(initial_state) # Or extract from the last event