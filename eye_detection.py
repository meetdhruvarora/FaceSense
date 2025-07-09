# import cv2
# import numpy as np
# import dlib

# # Load face detector and landmark predictor
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is available

# def head_pose_estimation(landmarks, size):
#     """
#     Estimates head pose based on the position of facial landmarks.
#     """
#     image_points = np.array([
#         (landmarks[30, 0], landmarks[30, 1]),  # Nose tip
#         (landmarks[8, 0], landmarks[8, 1]),    # Chin
#         (landmarks[36, 0], landmarks[36, 1]),  # Left eye left corner
#         (landmarks[45, 0], landmarks[45, 1]),  # Right eye right corner
#         (landmarks[48, 0], landmarks[48, 1]),  # Left Mouth corner
#         (landmarks[54, 0], landmarks[54, 1])   # Right mouth corner
#     ], dtype="double")

#     # 3D model points
#     model_points = np.array([
#         (0.0, 0.0, 0.0),            # Nose tip
#         (0.0, -330.0, -65.0),       # Chin
#         (-225.0, 170.0, -135.0),    # Left eye left corner
#         (225.0, 170.0, -135.0),     # Right eye right corner
#         (-150.0, -150.0, -125.0),   # Left Mouth corner
#         (150.0, -150.0, -125.0)     # Right mouth corner
#     ])

#     # Camera internals
#     focal_length = size[1]
#     center = (size[1] // 2, size[0] // 2)
#     camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

#     # Assume no lens distortion
#     dist_coeffs = np.zeros((4, 1))

#     # Solve for head pose
#     _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
#     return rotation_vector, translation_vector

# def detect_eyes_and_pose(video_file):
#     cap = cv2.VideoCapture(video_file)

#     distraction_count = 0
#     frame_count = 0
#     downward_looking_frames = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect face
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (fx, fy, fw, fh) in faces:
#             face = gray[fy:fy + fh, fx:fx + fw]
#             face_color = frame[fy:fy + fh, fx:fx + fw]

#             # Detect eyes
#             eyes = eye_cascade.detectMultiScale(face)
#             gaze_direction = "Looking Forward"  # Default gaze direction

#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#                 eye_region = face[ey:ey + eh, ex:ex + ew]
#                 _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
#                 contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
#                 for contour in contours:
#                     (x, y, w, h) = cv2.boundingRect(contour)
#                     if w > ew // 5 and h > eh // 5:
#                         pupil_center_x = x + w // 2

#                         # Determine gaze direction based on pupil position
#                         if pupil_center_x < ew // 4:  # Adjusted thresholds for better detection
#                             gaze_direction = "Looking Left"
#                         elif pupil_center_x > 3 * ew // 4:
#                             gaze_direction = "Looking Right"
#                         else:
#                             gaze_direction = "Looking Forward"

#                         # Display gaze direction on frame
#                         cv2.putText(frame, gaze_direction, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#                         # Count distractions
#                         if gaze_direction != "Looking Forward":
#                             distraction_count += 1
#                         frame_count += 1

#             # Detect facial landmarks for head pose estimation
#             dlib_rect = dlib.rectangle(fx, fy, fx + fw, fy + fh)
#             landmarks = predictor(gray, dlib_rect)
#             landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

#             # Get head pose
#             rotation_vector, translation_vector = head_pose_estimation(landmarks, frame.shape)

#             # Convert rotation vector to degrees for analysis
#             rvec_matrix, _ = cv2.Rodrigues(rotation_vector)
#             angles, _, _, _, _, _ = cv2.RQDecomp3x3(rvec_matrix)
            
#             pitch = angles[0] * 180 / np.pi  # Rotation around X-axis (Up-Down)
            
#             # If the pitch is above a certain threshold, mark as looking down
#             if pitch < -15:  # Adjust based on testing
#                 downward_looking_frames += 1

#         cv2.imshow('Eye and Head Pose Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     # Distracted behavior analysis
#     if frame_count > 0:
#         distraction_ratio = distraction_count / frame_count
#         if distraction_ratio > 0.3:  # Threshold: 30% of frames not looking forward
#             print("Candidate is distracted.")
#             return "Candidate is distracted"
#         else:
#             return "Candidate is Attentive"

#     # Cheating detection analysis
#     if downward_looking_frames > frame_count * 0.3:  # More than 30% of frames looking down
#         print("Candidate might be cheating.")
#         return "Candidate might be cheating"

# # # Example usage
# # video_file = r"C:\Users\dhruv\OneDrive\Pictures\Camera Roll\WIN_20241004_02_18_36_Pro.mp4"  # Replace with your video file path
# # detect_eyes_and_pose(video_file)
from fer import FER
from fer import Video
import pandas as pd
import cv2
import numpy as np
import dlib

# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

# Load face detector and landmark predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is available

def head_pose_estimation(landmarks, size):
    """
    Estimates head pose based on the position of facial landmarks.
    """
    image_points = np.array([
        (landmarks[30, 0], landmarks[30, 1]),  # Nose tip
        (landmarks[8, 0], landmarks[8, 1]),    # Chin
        (landmarks[36, 0], landmarks[36, 1]),  # Left eye left corner
        (landmarks[45, 0], landmarks[45, 1]),  # Right eye right corner
        (landmarks[48, 0], landmarks[48, 1]),  # Left Mouth corner
        (landmarks[54, 0], landmarks[54, 1])   # Right mouth corner
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left Mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve for head pose
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector, translation_vector

def detect_eyes_and_pose_and_emotion(video_file):
    cap = cv2.VideoCapture(video_file)

    distraction_count = 0
    frame_count = 0
    downward_looking_frames = 0
    positive_emotions = 0
    negative_emotions = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face for emotion recognition
        emotion_result = emotion_detector.detect_emotions(frame)
        if emotion_result:
            emotions = emotion_result[0]['emotions']
            positive_emotions += emotions['happy'] + emotions['surprise']
            negative_emotions += emotions['angry'] + emotions['disgust'] + emotions['fear'] + emotions['sad']

        # Detect face for eye and pose detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (fx, fy, fw, fh) in faces:
            face = gray[fy:fy + fh, fx:fx + fw]
            face_color = frame[fy:fy + fh, fx:fx + fw]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(face)
            gaze_direction = "Looking Forward"  # Default gaze direction

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                eye_region = face[ey:ey + eh, ex:ex + ew]
                _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if w > ew // 5 and h > eh // 5:
                        pupil_center_x = x + w // 2

                        # Determine gaze direction based on pupil position
                        if pupil_center_x < ew // 4:  # Adjusted thresholds for better detection
                            gaze_direction = "Looking Left"
                        elif pupil_center_x > 3 * ew // 4:
                            gaze_direction = "Looking Right"
                        else:
                            gaze_direction = "Looking Forward"

                        # Display gaze direction on frame
                        cv2.putText(frame, gaze_direction, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Count distractions
                        if gaze_direction != "Looking Forward":
                            distraction_count += 1
                        frame_count += 1

            # Detect facial landmarks for head pose estimation
            dlib_rect = dlib.rectangle(fx, fy, fx + fw, fy + fh)
            landmarks = predictor(gray, dlib_rect)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Get head pose
            rotation_vector, translation_vector = head_pose_estimation(landmarks, frame.shape)

            # Convert rotation vector to degrees for analysis
            rvec_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rvec_matrix)

            pitch = angles[0] * 180 / np.pi  # Rotation around X-axis (Up-Down)

            # If the pitch is above a certain threshold, mark as looking down
            if pitch < -15:  # Adjust based on testing
                downward_looking_frames += 1

        cv2.imshow('Emotion, Eye, and Head Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Emotion analysis
    if positive_emotions > negative_emotions:
        print("Person is interested based on emotions.")
    elif positive_emotions < negative_emotions:
        print("Person is not interested based on emotions.")
    else:
        print("Person has a neutral emotion.")

    # Distracted behavior analysis
    if frame_count > 0:
        distraction_ratio = distraction_count / frame_count
        if distraction_ratio > 0.3:
            if downward_looking_frames > frame_count * 0.3:  # More than 30% of frames looking down
                return("Candidate might be cheating.")  # Threshold: 30% of frames not looking forward
            return("Candidate is distracted.")
        else:
            return("Candidate is attentive.")

    # Cheating detection analysis
    

# # Example usage
# video_file = r"C:\Users\dhruv\OneDrive\Pictures\Camera Roll\WIN_20241004_02_18_36_Pro.mp4"  # Replace with your video file path
# detect_eyes_and_pose_and_emotion(video_file)
