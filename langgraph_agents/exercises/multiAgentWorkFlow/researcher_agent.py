from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools.web_scraper import scrape_web_page

# 1. Initialize the LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 2. Give it the specific tool it needs
researcher_tools = [scrape_web_page]

# 3. Create a strict system prompt
researcher_prompt = "You are the Lead Web Researcher. Use your scraping tools to extract live competitor land prices. Summarize your findings clearly and concisely."

# 4. Compile the subgraph
# This automatically handles the internal state, tool execution, and END routing
compiled_researcher = create_react_agent(
    model, 
    tools=researcher_tools, 
    state_modifier=researcher_prompt
)