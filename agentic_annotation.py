import os
import cv2
import json
from langgraph.graph import StateGraph, END

def ingestion_agent(state):
    print("[Ingestion] Loading raw videos...")
    video_dir = "/Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4"

    state["video_paths"] = [
        os.path.join(video_dir, f)

        for f in os.listdir(video_dir)
        if f.endswith(".mp4")
    ]

    return state

def frame_extraction_agent(state):
    print("[Frame Extraction] Extracting frames...")
    video_frames = {}

    for path in state["video_paths"]:
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        cap.release()
        video_frames[path] = frames
    
    state["video_frames"] = video_frames
    return state

def pose_estimation_agent(state):
    print("[Pose Estimation] (Stub) Generating keypoints...")
    keypoints = {}

    for path, frames in state["video_frames"].items():
        keypoints[path] = [{"keypoints": f"pose_{i}"} for i in range(len(frames))]
    state["keypoints"] = keypoints
    return state


def classify_movement(keypoint_sequence):
    if len(keypoint_sequence) == 0:
        return ["no_detection"]
    elif len(keypoint_sequence) > 60:
        return ["normal gait", "normal arm swing"]
    else:
        return ["reduced arm swing", "slowness"]
    
def labeling_agent(state):
    print("[Labeling] Inferring movement types...")
    labels = {}

    for path, keypoints in state["keypoints"].items():
        movement_labels = classify_movement(keypoints)
        labels[path] = movement_labels

    state["labels"] = labels
    return state

def annotation_agent(state):
    print("[Annotation] Saving annotated labels...")
    os.makedirs("data/annotated_data", exist_ok = True)
    with open("data/annotated_data/annotations.java", "w") as f:
        json.dump(state["labels"], f, indent=2)
    return state

# placeholder for fine-tuning Qwen2-VL
def training_agent(state):
    print("[Training] Fine-tuning Qwen2-VL (stub)...")

    return state

def evaluation_agent(state):
    print("[Evaluation] Prompting Qwen2-VL with clinical-style prompts (stub)...")

    return state


def run_pipeline():
    graph = StateGraph(state_schema=dict)
    graph.add_node("Ingest", ingestion_agent)
    graph.add_node("ExtractFrames", frame_extraction_agent)
    graph.add_node("PoseEstimation", pose_estimation_agent)
    graph.add_node("Labeling", labeling_agent)
    graph.add_node("Annotation", annotation_agent)
    graph.add_node("TrainModel", training_agent)
    graph.add_node("Evaluate", evaluation_agent)

    graph.set_entry_point("Ingest")

    graph.add_edge("Ingest", "Extract_Frames")
    graph.add_edge("Extract_Frames", "Pose_Estimation")
    graph.add_edge("PoseEstimation", "Labeling")
    graph.add_edge("Labeling", "Annotation")
    graph.add_edge("Annotation", "TrainModel")
    graph.add_edge("TrainModel", "Evaluate")
    
    graph.add_edge("Evaluate", END)

    workflow = graph.compile()
    initial_state = {}
    workflow.invoke(initial_state)

if __name__ == "__main__":
    run_pipeline()

