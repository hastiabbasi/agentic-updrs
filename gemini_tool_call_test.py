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

# dummy tool
@tool("get_pose_data", args_schema=PoseInput)
def get_pose_data(input: PoseInput) -> dict:
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