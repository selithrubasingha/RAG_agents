from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from tools.sql_query import query_database

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

analyst_tools = [query_database]
analyst_prompt = "You are the Senior Data Analyst. Query the SQL database to find internal land sales metrics. Do not make up data. If a query fails, adjust your SQL syntax and try again."

compiled_analyst = create_react_agent(
    model, 
    tools=analyst_tools, 
    prompt=analyst_prompt
)