import os
import cv2
import json
from langgraph.graph import StateGrpah, END

def ingestion_agent(state):
    print("[Ingestion] Loading raw videos...")
    video_dir = "/Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4"

    state["video_paths"] = [
        os.path.join(video_dir, f)

        for f in os.listdir(video_dir)
        if f.endswith(".mp4")
    ]

    return state