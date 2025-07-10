import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage