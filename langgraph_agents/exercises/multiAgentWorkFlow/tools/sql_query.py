# tools/sql_query.py
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
import os
# 1. Get the absolute path to the directory where THIS script lives (tools/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the multiAgentWorkFlow directory
parent_dir = os.path.dirname(script_dir)

# 3. Target the database file explicitly
db_path = os.path.join(parent_dir, "land_investment.db")
# Connect to your local database (e.g., SQLite for testing)
# Do NOT put this connection object inside the LangGraph State dictionary!
# Change this line:
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

@tool
def query_database(sql_query: str) -> str:
    """
    Executes a raw SQL query against the internal land investment database.
    Returns the raw tabular data results.
    """
    try:
        # db.run() executes the string and returns the row outputs
        result = db.run(sql_query)
        return result
    except Exception as e:
        # Returning the error as a string allows the LLM to read the error and try again
        return f"Database Execution Error: {str(e)}"