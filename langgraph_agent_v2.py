import os 
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, Optional, Dict, Any
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.pose_utils import extract_keypoints
import numpy as np

from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig

# import for visual representation of graph
from IPython.display import Image, display

# import for video input
from pydantic import BaseModel, Field

from scipy.signal import find_peaks 

# load env variables
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY not set in environment."

# replaced GraphState --> AgentState 
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    video_path: Optional[str]
    pose_data: Optional[any]
    velocity_data: Optional[Dict[str, float]]
    score_output: Optional[Dict[str, Any]]

class PoseInput(BaseModel):
    video_path: str = Field(description="The full local path to the video to analyze for UPDRS scoring.")

@tool("get_pose_data", args_schema=PoseInput)
def get_pose_data_tool(input: PoseInput) -> Dict:
    """Extracts RIGHT_INDEX and RIGHT_THUMB keypoints from a video using MediaPipe."""
    pose = extract_keypoints(input.video_path, joints=["RIGHT_INDEX", "RIGHT_THUMB"])
    # print(f"get_pose_data: extracted {len(pose)} frames")
    return {"pose_data": pose}

@tool 
def analyze_tap_amplitude(pose_data: list[Dict]) -> Dict:
    """Analyze tapping amplitude by measuring thumb-index distance per frame. Returns average amplitude and tap range."""

    distances = []

    for i, frame in enumerate(pose_data):
        if "RIGHT_INDEX" in frame and "RIGHT_THUMB" in frame:
            x1, y1 = frame["RIGHT_INDEX"]
            x2, y2 = frame["RIGHT_THUMB"]

            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(dist)

    if not distances:
        raise ValueError("No valid thumb-index distances found.")
    
    avg_amplitude = float(np.mean(distances))
    range_amplitude = float(np.max(distances) - np.min(distances))
    print(f"analyze_tap_amplitude: avg = {avg_amplitude:.4f}, range = {range_amplitude:.4f}")

    return {
        "avg_amplitude": avg_amplitude,
        "range_amplitude": range_amplitude 
    }
 

@tool
def score_updrs(avg_amplitude: float) -> dict:
    """Score UPDRS finger tapping based on the average thumb-index amplitude. Assume normalized coordinates."""

    if avg_amplitude > 0.04:
        score = 0
        rationale = "Normal tapping amplitude"
    elif avg_amplitude > 0.025: 
        score = 1
        rationale = "Slightly reduced amplitude"
    elif avg_amplitude > 0.015:
        score = 2 
        rationale = "Moderately reduced amplitude"
    else: 
        score = 3
        rationale = "Severely reduced amplitude"

    print(f"score_updrs: score = {score}, rationale = {rationale}")
    return {"score": score, "rationale": rationale, "avg_amplitude": avg_amplitude}


class TapFeatureInput(BaseModel):
    pose_data: list[dict]
    fps: int = Field(default = 30, description = "Frames per second of the video.")
    distance_threshold: float = Field(default = 0.01, description = "Minimum peak prominence to detect a tap.")

@tool("compute_tap_features", args_schema = TapFeatureInput)
def compute_tap_features(input: TapFeatureInput) -> dict:
    """
    Extracts high-level motion features from thumb-index distances in pose_data.

    Args:
        pose_data: List of frames with RIGHT_INDEX and RIGHT_THUMB keypoints.
        fps: Frames per second of the video.
        distance_threshold: Minimum distance change to count a tap.

    Returns: 
        Dictionary of clinically relevant tapping features.
    """

    pose_data = input.pose_data
    fps = input.fps
    distance_threshold = input.distance_threshold 

    # compute thumb-index distance per frame 
    distances = []

    for frame in pose_data:
        if "RIGHT_INDEX" in frame and "RIGHT_THUMB" in frame:
            x1, y1 = frame["RIGHT_INDEX"]
            x2, y2 = frame["RIGHT_THUMB"]
            dist = np.sqrt((x2 - x1)  ** 2 + (y2 - y1) ** 2)
            distances.append(dist)
    
    distances = np.array(distances)

    if len(distances) < 5:
        return {"Error": "Insufficient quantity of frames for valid analysis."}
    
    # normalize detect peaks (taps)
    norm_distances = distances - np.min(distances)
    norm_distances /= np.max(norm_distances) if np.max(norm_distances) != 0 else 1

    peaks, _ = find_peaks(norm_distances, distance = 3, prominence = distance_threshold)
    tap_count = len(find_peaks)

    # calculate tap frequency 
    duration_sec = len(distances) / fps
    tap_frequency_hz = tap_count / duration_sec if duration_sec > 0 else 0

    # amplitude stats
    amplitudes = norm_distances[peaks] if len(peaks) > 0 else []
    avg_tap_amplitude = float(np.mean(amplitudes)) if len(amplitudes) > 0 else 0.0
    amplitude_decrement_ratio = (
        float(amplitudes[-1] / amplitudes[0]) if len(amplitudes) >= 2 and amplitudes[0] !=0 else 1.0
    )

    # inter-tap variability 
    if len(peaks) >= 2:
        intertap_intervals = np.diff(peaks) / fps
        intertap_variability = float(np.std(intertap_intervals))
    else:
        intertap_variability = 0.0

    # rest time ratio
    rest_frames = np.sum(norm_distances < 0.1)
    rest_time_ratio = float(rest_frames / len(distances))

    return {
        "tap_count": tap_count,
        "tap_frequency_hz": round(tap_frequency_hz, 2),
        "avg_tap_amplitude": round(avg_tap_amplitude, 4),
        "amplitude_decrement_ratio": round(amplitude_decrement_ratio, 2),
        "intertap_variability": round(intertap_variability, 2),
        "rest_time_ratio": round(rest_time_ratio, 2),
        "frames_analyzed": len(distances),
    }
 

# tool bindings for LangGraph
tools = [get_pose_data_tool, compute_tap_features, score_updrs]
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
def call_model(state: AgentState, config: RunnableConfig) -> Dict:
    # use existing message chain
    response = model.invoke(state["messages"], config)
    print("tool_calls:", getattr(response, "tool_calls", None))
    print("Gemini message content: ", response.content)
    return {"messages": [response]}

def call_tool(state: AgentState) -> Dict:
    tool_outputs = []
    last_msg = state["messages"][-1]

    for call in getattr(last_msg, "tool_calls", []):
        tool = tools_by_name[call["name"]]

        try:
            result = tool.invoke(call["args"])
        except TypeError:
            # in case tool expects an input object instead of kwargs
            result = tool.func(PoseInput(**call["args"]))

        tool_outputs.append(ToolMessage(
            content = result, 
            name = call["name"],
            tool_call_id = call["id"]
        ))

    # debug statement 
    print(f"Calling {call['name']} with args: {call['args']}")
    return {"messages": tool_outputs}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]

    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("llm", call_model)
workflow.add_node("tools", call_tool)
workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue, {
    "tools": "tools",
    "end": END,
})
workflow.add_edge("tools", "llm")

graph = workflow.compile()

if __name__ == "__main__":
    
    video_path = "/Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4"
    prompt = f"Use get_pose_data with video_path=\"{video_path}\"" 
    inputs = {
        "messages": [HumanMessage(content=prompt)],
        "video_path": video_path
    }

    for step in graph.stream(inputs, stream_mode="values"):
        last = step["messages"][-1]
        print("\nStep: ")
        last.pretty_print()
    
    # visual representation of graph
    display(Image(graph.get_graph().draw_mermaid_png()))