import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# load API key
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "Missing GOOGLE_API_KEY in .env"

# default tool input schema 
class PoseInput(BaseModel):
    video_path: str = Field(description="Full local path to a Parkinson's finger tapping video.")