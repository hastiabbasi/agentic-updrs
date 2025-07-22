import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from pydantic import BaseModel, Field
from typing import dict

load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY not set in environment."

class PoseInput(BaseModel):
    video_path: str = Field(..., description="Local path to the UPDRS video.")

@tool("get_pose_data", args_schema=PoseInput)
def get_pose_data(input: PoseInput) -> dict:
    """Extracts RIGHT_INDEX and RIGHT_THUMB keypoints from a video using MediaPipe."""
    print(f"get_pose_data called on: {input.video_path}")
    # not actual values for now; test
    return {"pose_data": "simulated keypoint data"}

# registered the tool 
tools = [get_pose_data]
tools_by_name = {t.name: t for t in tools}

# gemini LLM + tool binding
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-pro",
    temperature = 0,
    convert_system_message_tohuman = True,
    streaming = False
)

def is_valid_message(msg: BaseMessage) -> bool:
    content = getattr(msg, "content", None)

    if isinstance(content, str):
        return content.strip() != ""
    if isinstance(content, dict):
        return any(v not in (None, "", {}, []) for v in content.values())
    return False