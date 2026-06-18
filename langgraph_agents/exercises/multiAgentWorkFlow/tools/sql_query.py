# tools/sql_query.py
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool

# Connect to your local database (e.g., SQLite for testing)
# Do NOT put this connection object inside the LangGraph State dictionary!
db = SQLDatabase.from_uri("sqlite:///land_investment.db")

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