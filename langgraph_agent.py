from typing import TypedDict, Optional, Dict, Any 
from langgraph.graph import StateGraph, END
from langchain.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np 