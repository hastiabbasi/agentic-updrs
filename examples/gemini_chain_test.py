import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from pydantic import BaseModel, Field
from typing import dict

load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY not set in environment."

class PoseINput(BaseModel):
    video_path: str = Field(..., description="Local path to the UPDRS video.")