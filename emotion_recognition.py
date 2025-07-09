from fer import FER
from fer import Video
import pandas as pd
emotion_detector = FER(mtcnn=True)
def emotion_recog(video_path):
    path_to_video = video_path
    # Define video 
    video = Video(path_to_video)
    result = video.analyze(emotion_detector, display=True)
    emotions_df = video.to_pandas(result)
    emotions_df.head()
    positive_emotions = sum(emotions_df.happy) + sum(emotions_df.surprise)
    negative_emotions = sum(emotions_df.angry) + sum(emotions_df.disgust) + sum(emotions_df.fear) + sum(emotions_df.sad)
    if positive_emotions > negative_emotions:
        print("Person is interested")
    elif positive_emotions < negative_emotions:
        print("Person is not interested")
    else:
        print("Person is neutral")
# import cv2
# from fer import FER
# import numpy as np

# # Initialize the FER emotion detector
# emotion_detector = FER(mtcnn=True)

# def emotion_recog_real_time():
#     # Open a connection to the webcam (0 for default webcam)
#     cap = cv2.VideoCapture(0)

#     positive_emotions = 0
#     negative_emotions = 0
#     frame_count = 0

#     while cap.isOpened():
#         # Capture each frame
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the frame to RGB as FER requires RGB input
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Detect emotions in the frame
#         emotion_result = emotion_detector.detect_emotions(rgb_frame)
        
#         # If an emotion is detected, process it
#         if emotion_result:
#             emotions = emotion_result[0]['emotions']  # Get the emotion dictionary from the result

#             # Display the emotion scores on the frame
#             text = f"Happiness: {emotions['happy']:.2f}, Anger: {emotions['angry']:.2f}, Sad: {emotions['sad']:.2f}, Surprise: {emotions['surprise']:.2f}"
#             cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # Add to emotion counts for analysis
#             positive_emotions += emotions['happy'] + emotions['surprise']
#             negative_emotions += emotions['angry'] + emotions['disgust'] + emotions['fear'] + emotions['sad']
#             frame_count += 1

#         # Display the frame with detected emotions
#         cv2.imshow('Real-time Emotion Recognition', frame)

#         # Press 'q' to quit the real-time emotion detection
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture object and close any open windows
#     cap.release()
#     cv2.destroyAllWindows()

#     # Analyze the overall emotions based on the captured frames
#     if frame_count > 0:
#         if positive_emotions > negative_emotions:
#             print("Person is interested")
#         elif positive_emotions < negative_emotions:
#             print("Person is not interested")
#         else:
#             print("Person is neutral")

# # Call the real-time emotion recognition function
# emotion_recog_real_time()
