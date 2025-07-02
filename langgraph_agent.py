from typing import TypedDict, Optional, Dict, Any 
from langgraph.graph import StateGraph, END
from langchain.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np 

# pose extraction tool 
from utils.pose_utils import extract_keypoints

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    convert_system_message_to_human=True
)

class GraphState(TypedDict):
    user_input: str 
    video_path: str
    pose_data: Optional[Any]
    velocity_data: Optional[Dict[str, float]]
    tremor_data: Optional[Dict[str, float]]
    tremor_data: Optional[Dict[str, Any]]

@tool 
def extract_pose_node(state: GraphState) -> GraphState:
    pose = extract_keypoints(state['video_path'], joints=["RIGHT_INDEX", "RIGHT_THUMB"])
    return {**state, "pose_data": pose}