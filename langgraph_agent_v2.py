import os 
from dotenv import load_dotenv
from typing import TypedDict, Optional, Dict, Any
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.pose_utils import extract_keypoints
import numpy as np

# load env variables
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY not set in environment."

# define state
class GraphState(TypedDict):
    user_input: str
    video_path: str
    pose_data: Optional[Any]
    velocity_data: Optional[Dict[str, float]]
    score_output: Optional[dict[str, Any]]

@tool
def get_pose_data(video_path: str) -> Dict[str, Any]:
    """Extracts RIGHT_INDEX and RIGHT_THUMB keypoints from a video using MediaPipe."""
    pose = extract_keypoints(video_path, joints=["RIGHT_INDEX", "RIGHT_THUMB"])
    print(f"get_pose_data: extracted {len(pose)} frames")
    return {"pose_data": pose}