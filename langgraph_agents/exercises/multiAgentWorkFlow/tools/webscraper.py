# tools/web_scraper.py
import os
from langchain_core.tools import tool
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def scrape_web_page(url: str) -> str:
    """
    Extracts and cleans raw webpage content into readable markdown.
    Use this to read specific competitor pricing pages or land registry URLs.
    """
    try:
        # The extract API natively handles JavaScript rendering and HTML stripping
        extract_response = tavily_client.get_extract(urls=[url])
        
        # Safely extract the raw markdown content
        content = extract_response.get("results", [])[0].get("raw_content", "")
        
        if not content:
             return "Scraping Error: The page returned empty content."
             
        return content
    except Exception as e:
        return f"Scraping Error: Failed to connect to {url}. Details: {str(e)}"