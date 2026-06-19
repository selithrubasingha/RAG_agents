

import os
from pathlib import Path
from typing import Annotated, List, Sequence, TypedDict , Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader  # document_loader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from pydantic import BaseModel, Field
from tavily import TavilyClient
# --- 1. GEMINI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()  # Load environment variables from .env file
