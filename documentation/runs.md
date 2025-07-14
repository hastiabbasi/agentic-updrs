run #1: 
================================ Human Message =================================

Extract the pose and score the UPDRS finger tapping for this video: /Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4
/Users/hastiabbasi/agentic-updrs/agentic-updrs/.venv/lib/python3.10/site-packages/langchain_google_genai/chat_models.py:357: UserWarning: Convert_system_message_to_human will be deprecated!
  warnings.warn("Convert_system_message_to_human will be deprecated!")
Step: 
================================== Ai Message ==================================

Please provide the path to the video file you would like me to analyze.

run #2:
Step: 
================================ Human Message =================================

Use get_pose_data with video_path="/Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4"
/Users/hastiabbasi/agentic-updrs/agentic-updrs/.venv/lib/python3.10/site-packages/langchain_google_genai/chat_models.py:357: UserWarning: Convert_system_message_to_human will be deprecated!
  warnings.warn("Convert_system_message_to_human will be deprecated!")
tool_calls: [{'name': 'get_pose_data', 'args': {'video_path': '/Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4'}, 'id': '*redacted*', 'type': 'tool_call'}]
Gemini message content:  

Step: 
================================== Ai Message ==================================
Tool Calls:
  get_pose_data (*redacted*)
 Call ID: *redacted*
  Args:
    video_path: /Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
GL version: 2.1 (2.1 Metal - 89.3), renderer: Apple M2
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
get_pose_data: extracted 1000 frames
Calling get_pose_data with args: {'video_path': '/Users/hastiabbasi/agentic-updrs/agentic-updrs/FT_vids/sub1vid7.mp4'}

Step: 
================================= Tool Message =================================
Name: get_pose_data

{'pose_data': [{'RIGHT_INDEX': (0.6105872392654419, 0.6199342608451843), 'RIGHT_THUMB': (0.5951260328292847, 0.6046525835990906)}, {'RIGHT_INDEX': (0.6097170114517212, 0.6212522983551025), 'RIGHT_THUMB': (0.5948037505149841, 0.6056723594665527)}, {'RIGHT_INDEX': (0.6097207069396973, 0.6223471760749817), 'RIGHT_THUMB': (0.5950323343276978, 0.6067273020744324)}, {'RIGHT_INDEX': (0.6098368763923645, 0.6230593323707581), 'RIGHT_THUMB': (0.5953595042228699, 0.6074672937393188)}, {'RIGHT_INDEX': (0.6099581122398376, 0.623433530330658), 'RIGHT_THUMB': (0.5955961346626282, 0.6078372597694397)}, {'RIGHT_INDEX': (0.6099935173988342, 0.6234226226806641), 'RIGHT_THUMB': (0.5957047939300537, 0.6078282594680786)}, {'RIGHT_INDEX': (0.6101094484329224, 0.6234089732170105), 'RIGHT_THUMB': (0.5959474444389343, 0.607807457447052)}, {'RIGHT_INDEX': (0.6101158857345581, 0.6233766078948975), 'RIGHT_THUMB': (0.5959879159927368, 0.6077160835266113)}, {'RIGHT_INDEX': (0.6101427674293518, 0.6233466267585754), 'RIGHT_THUMB': (0.5960482954978943, 0.607597827911377)}, 

*max frames set to 1000 made pose_data immensely large --> interrepted run adjust input*