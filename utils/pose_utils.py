import mediapipe as mp
import cv2

def extract_keypoints(video_path: str, joints: list = None, max_frames: int = 10) -> list:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)

    output = []
    frame_count = 0 
    joint_set = set(joints) if joints else None

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()

        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks: 
            landmarks = results.pose_landmarks.landmark
            frame_keypoints = {}

            for idx, lm in enumerate(landmarks):
                joint_name = mp_pose.PoseLandmark(idx).name
                
                if joint_set is None or joint_name in joint_set:
                    frame_keypoints[joint_name] = (lm.x, lm.y)

            output.append(frame_keypoints)

        frame_count += 1

    cap.release()
    return output