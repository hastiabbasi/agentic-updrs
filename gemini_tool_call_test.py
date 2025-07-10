import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from langchain_core.messages import ToolMessage 

# load API key
load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "Missing GOOGLE_API_KEY in .env"

# default tool input schema 
class PoseInput(BaseModel):
    video_path: str = Field(description="Full local path to a Parkinson's finger tapping video.")

# dummy tool
@tool("get_pose_data", args_schema=PoseInput)
def get_pose_data(input: PoseInput) -> dict:
    """Extracts RIGHT_INDEX and RIGHT_THUMB keypoints from a video using MediaPipe."""
    print(f"get_pose_data() called with {input.video_path}")
    return{"message": f"Processed video at {input.video_path}"}

# bind tool to gemini 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.0,
    convert_system_message_to_human=True,
    streaming=False
)

# tool binding 
model = llm.bind_tools([get_pose_data])

# test prompt 
video_path = "/Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4"
user_prompt = f"Please run get_pose_data with video_path: {video_path}"

# send prompt
messages = [HumanMessage(content=user_prompt)]
response = model.invoke(messages)

# output 
print("\n AI Message: ", response.content)
print("tool_calls: ", getattr(response, "tool_calls", None))

tool_call = response.tool_calls[0]

# call the actual tool
tool_result = get_pose_data.invoke(tool_call["args"])

# return the result as a ToolMessage so Gemini can reason on it 
tool_msg = ToolMessage(
    content=tool_result,
    name=tool_call["name"],
    tool_call_id=tool_call["id"]
)

'''
New message list includes:
1) Original user prompt
2) Gemini's tool call
3) Our ToolMessage response
'''
new_messages = [
    HumanMessage(content=user_prompt),
    # Gemini's tool call message
    response,
    # tool's result
    tool_msg
]

followup_response = model.invoke(new_messages)

print("\n Gemini follow-up:", followup_response.content)

