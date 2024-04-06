import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

TEXT_COLOR = (0, 255, 0)
FONT_SIZE = 1.5
OVERLAY_COLOR_POINT = (230, 230, 230)
OVERLAY_COLOR_LINES = (200, 200, 200)

# Landmark indices for eyebrows, mouth, mouth openness, and eyes
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_INDICES = [336, 296, 334, 293, 300]
MOUTH_INDICES = [61, 291, 0, 17, 269, 405]
MOUTH_OPENNESS_INDICES = [13, 14, 78, 308, 402, 311, 312, 87, 178, 88, 95, 78]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_eyebrow_extension(landmarks, eyebrow_indices):
    eyebrow_points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in eyebrow_indices])
    eyebrow_center = np.mean(eyebrow_points, axis=0)
    eyebrow_distances = np.linalg.norm(eyebrow_points - eyebrow_center, axis=1)
    max_distance = np.max(eyebrow_distances)
    extension = max_distance / 0.05  # Normalize the extension
    return np.clip(extension, 0.0, 1.0)

def calculate_smiledness(landmarks, mouth_indices):
    SMILE_GAIN = 5
    mouth_points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in mouth_indices])
    mouth_center = np.mean(mouth_points, axis=0)
    mouth_distances = np.linalg.norm(mouth_points - mouth_center, axis=1)
    max_distance = np.max(mouth_distances)
    smiledness = (max_distance - 0.02) / 0.06  # Normalize the smiledness
    return np.clip((smiledness * 2 - 1) * SMILE_GAIN, -1.0, 1.0)

def calculate_smiledness_2(landmarks, mouth_indices):
    SMILE_GAIN = 5
    mouth_points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in mouth_indices])
    mouth_center = np.mean(mouth_points, axis=0)
    mouth_distances = np.linalg.norm(mouth_points - mouth_center, axis=1)
    max_distance = np.max(mouth_distances)
    smiledness = (max_distance - 0.02) / 0.06  # Normalize the smiledness
    return np.clip((smiledness * 2 - 1) * SMILE_GAIN, -1.0, 1.0)

def calculate_mouth_openness(landmarks, mouth_openness_indices):
    mouth_points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in mouth_openness_indices])
    upper_lip_points = mouth_points[:6]
    lower_lip_points = mouth_points[6:]
    upper_lip_center = np.mean(upper_lip_points, axis=0)
    lower_lip_center = np.mean(lower_lip_points, axis=0)
    mouth_openness = np.linalg.norm(upper_lip_center - lower_lip_center)
    return np.clip(mouth_openness / 0.1, 0.0, 1.0)

def calculate_mouth_openness_2(landmarks, mouth_openness_indices):
    upper_lip_center = landmarks[13]
    lower_lip_center = landmarks[14]
    mouth_open_size = upper_lip_center.y - lower_lip_center.y
    face_size = landmarks[10].y - landmarks[175].y
    MAX_MOUTH_OPEN_SIZE = 0.15 * face_size
    mouth_openness = mouth_open_size / MAX_MOUTH_OPEN_SIZE
    return np.clip(mouth_openness, 0.0, 1.0)

def calculate_eye_openness(landmarks, eye_indices):
    eye_points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices])
    eye_vertical_indices = [1, 2, 4, 5]
    eye_vertical_points = eye_points[eye_vertical_indices]
    eye_height = np.linalg.norm(eye_vertical_points[0] - eye_vertical_points[1]) + \
                 np.linalg.norm(eye_vertical_points[2] - eye_vertical_points[3])
    eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
    eye_openness = eye_height / (eye_width)
    return np.clip(eye_openness, 0.0, 1.0)

def calculate_eye_openness_2(landmarks, eye_indices):
    eye_points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices])
    eye_vertical_indices = [1, 2, 4, 5]
    eye_vertical_points = eye_points[eye_vertical_indices]
    eye_height = np.linalg.norm(eye_vertical_points[0] - eye_vertical_points[1]) + \
                 np.linalg.norm(eye_vertical_points[2] - eye_vertical_points[3])
    eye_width = np.linalg.norm(eye_points[0] - eye_points[3])

    left_eye_height = landmarks[144].y - landmarks[160].y # ~= landmarks[158].y - landmarks[153].y
    left_eye_width = landmarks[158].x - landmarks[160].x # ~= 153.x - 144.x

    # face_size = landmarks[10].y - landmarks[175].y
    # MAX_EYE_OPEN_SIZE = 0.02 * face_size

    # eye_openness = left_eye_height / left_eye_width
    # eye_openness = left_eye_height / MAX_EYE_OPEN_SIZE
    # return np.clip(eye_openness, 0.0, 1.0)
    return landmarks[144].y

# Initialize MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    # Start video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = face_mesh.process(rgb_frame)


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=DrawingSpec(color=OVERLAY_COLOR_POINT, thickness=1, circle_radius=1),
                    connection_drawing_spec=DrawingSpec(color=OVERLAY_COLOR_LINES, thickness=1)
                )

            for face_landmarks in results.multi_face_landmarks:
                # Calculate eyebrow extension for left and right eyebrows
                left_extension = calculate_eyebrow_extension(face_landmarks.landmark, LEFT_EYEBROW_INDICES)
                right_extension = calculate_eyebrow_extension(face_landmarks.landmark, RIGHT_EYEBROW_INDICES)

                # Calculate smiledness
                smiledness = calculate_smiledness(face_landmarks.landmark, MOUTH_INDICES)

                # Calculate mouth openness
                # mouth_openness = calculate_mouth_openness(face_landmarks.landmark, MOUTH_OPENNESS_INDICES)
                mouth_openness = calculate_mouth_openness_2(face_landmarks.landmark, MOUTH_OPENNESS_INDICES)

                # Calculate eye openness for left and right eyes
                # left_eye_openness = calculate_eye_openness(face_landmarks.landmark, LEFT_EYE_INDICES)
                # right_eye_openness = calculate_eye_openness(face_landmarks.landmark, RIGHT_EYE_INDICES)
                left_eye_openness = calculate_eye_openness_2(face_landmarks.landmark, LEFT_EYE_INDICES)
                right_eye_openness = calculate_eye_openness_2(face_landmarks.landmark, RIGHT_EYE_INDICES)

                # Draw the eyebrow extension, smiledness, mouth openness, and eye openness values on the frame
                cv2.putText(frame, f"Left Eyebrow Extension: {left_extension:.1f}", (10, int(30 * FONT_SIZE)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 2)
                cv2.putText(frame, f"Right Eyebrow Extension: {right_extension:.1f}", (10, int(60 * FONT_SIZE)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 2)
                cv2.putText(frame, f"Smiledness: {smiledness:.1f}", (10, int(90 * FONT_SIZE)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 2)
                cv2.putText(frame, f"Mouth Openness: {mouth_openness:.1f}", (10, int(120 * FONT_SIZE)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 2)
                # cv2.putText(frame, f"Left Eye Openness: {left_eye_openness:.1f}", (10, int(150 * FONT_SIZE)),
                cv2.putText(frame, f"Left Eye Openness: {left_eye_openness}", (10, int(150 * FONT_SIZE)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 2)
                cv2.putText(frame, f"Right Eye Openness: {right_eye_openness:.1f}", (10, int(180 * FONT_SIZE)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 2)

        # Display the frame
        cv2.imshow('Facial Analysis', frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()