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

model = llm.bind_tools(tools)

def is_valid_message(msg: BaseMessage) -> bool:
    content = getattr(msg, "content", None)

    if isinstance(content, str):
        return content.strip() != ""
    if isinstance(content, dict):
        return any(v not in (None, "", {}, []) for v in content.values())
    return False

def call_model(messages: list[BaseMessage]):
    clean_messages = [msg for msg in messages if is_valid_message(msg)]

    print("Cleaned messages:")
    
    for i, msg in enumerate(clean_messages):
        print(f"    [{i}] {type(msg).__name__}: {repr(msg.content)}")

    response = model.invoke(clean_messages)

    print("\n Gemini Response:", response.content)
    print("tool_calls:", getattr(response, "tool_calls", None))

    return response

if __name__ == "__main__":
    video_path = "/Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4"
    prompt = f"Analyze this video: {video_path}"
    messages = [HumanMessage(content=prompt)]

    # trigger tool call
    ai_response = call_model(messages)

    new_messages = [messages[0], ai_response]

    for call in getattr(ai_response, "tool_calls", []):
        tool = tools_by_name[call["name"]]
        result = tool.invoke(tool.args_schema(**call["args"]))
        print(f"Tool result: {result}")
        msg = ToolMessage(
            content = result, 
            name = call["name"],
            tool_call_id = call["id"]
        )

        new_messages.append(msg)

    if len(new_messages) > 2:
        print("\n Sending tool result back to Gemini")
        final_response = call_model(new_messages)