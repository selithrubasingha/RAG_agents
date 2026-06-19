import asyncio
from typing import TypedDict, List, Union
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from livekit import agents, rtc
from livekit.agents import AgentSession, Agent
from livekit.plugins import google, deepgram, elevenlabs

load_dotenv()

# --- 1. LANGGRAPH (The Brain) ---
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7) 

async def process(state: AgentState) -> AgentState:
    response = await llm.ainvoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent_brain = graph.compile()


# --- 2. LIVEKIT (The Ears and Mouth) ---
async def entrypoint(ctx: agents.JobContext):
    """This function replaces your while loop. It runs when someone joins the voice room."""
    
    # Connect to the LiveKit room
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    print("✅ Agent connected to LiveKit room!")

    # NEW v1.0+ FRAMEWORK: VoicePipelineAgent is completely gone. We use AgentSession now.
    session = AgentSession(
        stt=deepgram.STT(),
        llm=google.LLM(model="gemini-2.5-flash"), # Keeping Google LLM as your placeholder
        tts=elevenlabs.TTS()
    )

    # Define the agent's core instructions
    my_agent = Agent(
        instructions="You are an intelligent land investment agent."
    )

    # Start the session
    await session.start(agent=my_agent, room=ctx.room)
    
    # Tell the TTS to speak the greeting aloud
    await session.say("Hello, I am your intelligent land investment agent. How can I help you?")

if __name__ == "__main__":
    import asyncio
    
    # Patch for Python 3.14 compatibility with LiveKit CLI
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))