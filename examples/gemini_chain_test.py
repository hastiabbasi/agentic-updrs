import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from pydantic import BaseModel, Field
from typing import dict