import cv2
import mediapipe as mp
import math

# Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

GOLDEN_RATIO = 1.618

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def draw_ratio_text(img, label, ratio, ideal, x, y):
    color = (0, 255, 0) if abs(ratio - ideal) < 0.1 else (0, 0, 255)
    cv2.putText(img, f'{label}: {ratio:.2f} (Ideal: {ideal})', (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            lm = face_landmarks.landmark

            # Landmarks
            forehead = lm[10]
            chin = lm[152]
            left_cheek = lm[234]
            right_cheek = lm[454]
            left_eye_outer = lm[33]
            right_eye_outer = lm[263]
            left_eye_inner = lm[133]
            right_eye_inner = lm[362]
            nose_tip = lm[1]
            nose_base = lm[2]
            nose_left = lm[94]
            nose_right = lm[331]
            mouth_left = lm[78]
            mouth_right = lm[308]
            mouth_top = lm[13]
            mouth_bottom = lm[14]
            brow_mid = lm[168]

            # Measurements
            face_length = euclidean(forehead, chin)
            face_width = euclidean(left_cheek, right_cheek)
            eye_width = euclidean(left_eye_outer, right_eye_outer)
            inter_eye_dist = euclidean(left_eye_inner, right_eye_inner)
            nose_width = euclidean(nose_left, nose_right)
            mouth_width = euclidean(mouth_left, mouth_right)
            nose_to_chin = euclidean(nose_base, chin)
            brow_to_nose = euclidean(brow_mid, nose_base)
            nose_to_mouth = euclidean(nose_base, mouth_top)
            mouth_to_chin = euclidean(mouth_bottom, chin)

            # Ratios
            ratios = {
                "Face L/W": face_length / face_width if face_width else 0,
                "Eye/Nose Width": eye_width / nose_width if nose_width else 0,
                "Inter-Eye/Eye Width": inter_eye_dist / eye_width if eye_width else 0,
                "Nose/Mouth Width": nose_width / mouth_width if mouth_width else 0,
                "Upper Face / Lower Face": brow_to_nose / nose_to_chin if nose_to_chin else 0,
                "Mouth to Chin / Nose to Mouth": mouth_to_chin / nose_to_mouth if nose_to_mouth else 0,
            }

            # Display ratios
            y_offset = 30
            for name, ratio in ratios.items():
                draw_ratio_text(frame, name, ratio, GOLDEN_RATIO, 20, y_offset)
                y_offset += 30

    cv2.imshow("Perfect Face Ratio Analyzer", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()





# # project: Face Recognition
# import cv2
# import face_recognition
# import pickle
# import os

# ENCODING_FILE = "encodings.pickle"

# def save_encoding(face_encoding):
#     """Save the face encoding to a file"""
#     with open(ENCODING_FILE, "wb") as f:
#         pickle.dump(face_encoding, f)

# def load_encoding():
#     """Load the face encoding from file"""
#     if os.path.exists(ENCODING_FILE):
#         with open(ENCODING_FILE, "rb") as f:
#             return pickle.load(f)
#     return None

# # Check if an encoding exists
# stored_encoding = load_encoding()

# if stored_encoding is None:
#     print("No saved face detected! Capturing your face...")

#     video_capture = cv2.VideoCapture(0)

#     while True:
#         ret, frame = video_capture.read()
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
#         if face_encodings:
#             print("Face captured and saved!")
#             save_encoding(face_encodings[0])
#             stored_encoding = face_encodings[0]
#             break

#         cv2.imshow("Face Capture", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()

# # Start real-time face recognition
# video_capture = cv2.VideoCapture(0)

# while True:
#     ret, frame = video_capture.read()
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         match = face_recognition.compare_faces([stored_encoding], face_encoding)
#         name = "Unknown"

#         if match[0]:
#             name = "Yuvaraj Raju"
        

#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     cv2.imshow("Real-Time Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()






# # import cv2
# # import face_recognition

# # # Open webcam
# # video_capture = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = video_capture.read()
# #     if not ret:
# #         break

# #     # Convert frame from BGR (OpenCV) to RGB (face_recognition)
# #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     # Detect face locations and landmarks
# #     face_locations = face_recognition.face_locations(rgb_frame)
# #     face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

# #     for face_landmarks in face_landmarks_list:
# #         for facial_feature in face_landmarks.keys():
# #             points = face_landmarks[facial_feature]

# #             # Draw lines between points
# #             for i in range(len(points) - 1):
# #                 cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

# #             # Draw points
# #             for point in points:
# #                 cv2.circle(frame, point, 2, (0, 0, 255), -1)  # Red points

# #     # Show the output
# #     cv2.imshow("Face Landmarks (Points & Lines)", frame)

# #     # Press 'q' to exit
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release the camera and close windows
# # video_capture.release()
# # cv2.destroyAllWindows()
