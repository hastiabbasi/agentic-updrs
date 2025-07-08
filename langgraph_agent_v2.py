import os 
from dotenv import load_dotenv
from typing import TypedDict, Optional, Dict, Any
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.pose_utils import extract_keypoints
import numpy as np

from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from langchain_core.runnables import RunnableConfig

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

@tool 
def analyze_finger_velocity(pose_data: Any) -> Dict[str, float]:
    """Computes average tapping velocity from pose keypoints."""
    velocities = []
    for i in range(1, len(pose_data)):
        prev = pose_data[i - 1]
        curr = pose_data[i]

        if "RIGHT_INDEX" in prev and "RIGHT_INDEX" in curr:
            dx = curr["RIGHT_INDDEX"][0] - prev["RIGHT_INDEX"][0]
            dy = curr["RIGHT_INDEX"][1] - prev["RIGHT_INDEX"][1]
            velocities.append(np.sqrt(dx ** 2 + dy ** 2) * 30)
        
    avg_velocity = float(np.mean(velocities)) if velocities else 0.0
    print(f"analyze_finger_velocity: avg_velocity = {avg_velocity:.4f}")
    return {"avg_velocity": avg_velocity}

@tool
def score_updrs(avg_velocity: float) -> Dict[str, Any]:
    """Scores UPDRS Finger Tapping task based on average velocity."""
    if avg_velocity > 1.5:
        score = 0
        rationale = "Normal tapping speed"
    elif avg_velocity > 1.0:
        score = 1
        rationale = "Slight slowing"
    elif avg_velocity > 0.5:
        score = 2
        rationale = "Moderate slowing"
    else:
        score = 3
        rationale = "Sever bradykinesia"

    print(f"score_updrs: score = {score}, rationale = {rationale}")
    return {"score": score, "rationale": rationale, "velocity": avg_velocity}

# tool bindings for LangGraph
tools = [get_pose_data, analyze_finger_velocity, score_updrs]
tools_by_name = {t.name: t for t in tools}

# initialize gemini 
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-pro",
    temperature = 0.2,
    convert_system_message_to_human=True,
    streaming=False
)

model = llm.bind_tools(tools)

# LangGraph nodes
def call_model(state: GraphState, config: RunnableConfig) -> Dict:
    user_msg = HumanMessage(content=state["user_input"])
    response = model.invoke([user_msg], config)
    return {"messages": [response]}

def call_tool(state: GraphState) -> Dict:
    messages = state.get("messages", [])
    last_msg = messages[-1]
    tool_outputs = []

    for call in last_msg.tool_calls:
        tool = tools_by_name[call["name"]]
        result = tool.invoke(call["args"])
        tool_outputs.append(ToolMessage(content=result, name=call["name"], tool_call_id=call["id"]))
    
    return {"messages": tool_outputs}

def should_continue(state: GraphState) -> str:
    messages = state.get("messages", [])

    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    return "end"