from typing import TypedDict, Optional, Dict, Any 
from langgraph.graph import StateGraph, END
from langchain.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np 

from utils.pose_utils import extract_keypoints
# from tools.tremor import analyze_tremor

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

@tool 
def analyze_velocity_node(state: GraphState) -> GraphState:
    keypoints = state['pose_data']
    velocities = []
    for i in range(1, len(keypoints)):
        prev, curr = keypoints[i-1], keypoints[i]
        if "RIGHT_INDEX" in prev and "RIGHT_INDEX" in curr:
            dx = curr["RIGHT_INDEX"][0] - prev["RIGHT_INDEX"][0]
            dy = curr["RIGHT_INDEX"][1] - prev["RIGHT_INDEX"][1]
            velocities.append(np.sqrt(dx**2 + dy**2) * 30)
        mean_vel = float(np.mean(velocities)) if velocities else 0.0
        return {**state, "velocity_data": {"avg_velocity": mean_vel}}
    
@tool
def score_finger_tap_node(state: GraphState) -> GraphState:
    velocity = state["velocity_data"]["avg_velocity"]

    if velocity > 1.5:
        score = 0
        rationale = "Normal tapping speed"
    elif velocity > 1.0:
        score = 1
        rationale = "Slight slowing"
    elif velocity > 0.5:
        score = 2 
        rationale = "Moderate slowing"
    else:
        score = 3
        rationale = "Severe bradykinesia"
    return {**state, "score_output": {"score": score, "rationale": rationale, "velocity": velocity}}

@tool 
def tremor_node(state: GraphState) -> GraphState:
    tremor = analyze_tremor(state["video_path"])
    return {**state, "tremor_data": tremor}