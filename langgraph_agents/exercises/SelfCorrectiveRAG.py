from imports import *


load_dotenv()


class AgentState(TypedDict):
    question : str
    response : str
    document :List[Document] 
    hallucinate : str
    feedback : str
class HallucinationCheck(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    
    hallucinate: Literal["yes", "no"] = Field(
        ...,
        description="Does the LLM's response contain hallucinations? Does the response contain false and misleading information that is not contained the document itself? Answer 'yes' or 'no'."
    )
    
tools = []
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
    max_retries=3 # Built-in protection against connection drops
).bind_tools(tools)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def document_storage(relative_path: str):
    """
    Loads a PDF from a relative path string, splits it, 
    and indexes it into an InMemoryVectorStore.
    """
    # 1. Resolve the absolute path relative to where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.abspath(os.path.join(script_dir, relative_path))

    # 2. Safety check: Verify the file actually exists
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Could not find document at: {absolute_path}")

    try:
        # 3. Load the document directly from the file system
        loader = PyPDFLoader(absolute_path)
        docs = loader.load()

        # 4. Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        # 5. Initialize and populate the InMemoryVectorStore
        # This replaces FAISS completely and runs entirely in RAM
        vector_store = InMemoryVectorStore.from_documents(
            documents=all_splits, 
            embedding=embeddings
        )

        return vector_store

    except Exception as e:
        raise RuntimeError(f"Failed to process and index document: {e}")


document = document_storage("sample.pdf")  # Adjust the path as needed

def retrieve_relevant_docs(state: AgentState) -> AgentState:
    """
    Retrieves the top-k most relevant documents from the vector store based on the query.
    """
    try:
        results = document.similarity_search(state['question'])
        state['document'] = results
        return state
    except Exception as e:
        raise RuntimeError(f"Error during document retrieval: {e}")
    
def generate_response(state: AgentState) -> AgentState:
    """
    Generates a response based on the user's question and the retrieved documents.
    """
    try:
     # Inside your generate_response node function:
        relevant_docs = state['document']
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Dynamically append instructions if we are on a retry run
        feedback_guideline = f"\n\nCorrection Guideline: {state['feedback']}" if state.get("feedback") else ""

        prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {state['question']}{feedback_guideline}"
        # Generate the response using the LLM
        response = model.invoke([HumanMessage(content=prompt)])
        state['response'] = response.content

        return state
    except Exception as e:
        raise RuntimeError(f"Error during response generation: {e}")

def did_hallucinate(state: AgentState) -> AgentState:
    """
    Evaluates the LLM's response for hallucinations against the retrieved documents.
    """
    system_prompt = """You are an expert data auditor grading LLM responses for hallucinations in a Land Investment System.
    Analyze the provided response and the retrieved context carefully. Determine if the response contains any false or misleading information that is not supported by the retrieved documents.
    Provide a strict binary score: return 'yes' if the response contains hallucinations, or 'no' if it is fully supported by the retrieved context."""

    structured_llm = model.with_structured_output(HallucinationCheck)

    retrieved_chunks = "\n\n".join([doc.page_content for doc in state["document"]])
    llm_response = state["response"]

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{retrieved_chunks}\n\nLLM Response:\n{llm_response}")
    ]
    
    result = structured_llm.invoke(messages)
    
    state["hallucinate"] = result.hallucinate

    if result.hallucinate == "yes":
        state["feedback"] = "CRITICAL: Your previous response contained claims not supported by the document. Restrict your next answer strictly to the provided text context."
    else:
        state["feedback"] = ""
    return state   

def should_continue(state: AgentState) -> str:
    if state["hallucinate"] == "yes":
        return "agent"
    else:
        return "end" 
graph = StateGraph(AgentState)

graph.add_node("retrieve_docs", retrieve_relevant_docs)
graph.add_node("generate_response", generate_response)
graph.add_node("did_hallucinate", did_hallucinate)

graph.set_entry_point("retrieve_docs")
graph.add_edge("retrieve_docs", "generate_response")
graph.add_edge("generate_response", "did_hallucinate")

graph.add_conditional_edges("did_hallucinate",
                             should_continue,
                               {"agent": "generate_response", 
                                "end": END
                                }
                                ) 
app = graph.compile()




if __name__ == "__main__":
    print("\n--- LangGraph RAG Agent Initialized ---")
    
    user_query = "What is the main topic of the document?"
    
    initial_state = {
        "question": user_query,
        "response": "",      
        "document": [],
        "hallucinate": "",
        "feedback": ""       
    }
    
    print(f"Question: {user_query}")
    print("Running graph...\n")
    
    try:
        final_state = app.invoke(initial_state)
        
        print("--- Final Output ---")
        print(final_state["response"])
        
    except Exception as e:
        print(f"Graph execution failed: {e}")

