import cv2

from app.model.detectors.blink_detector import extract_eye_landmarks, calculate_ear, draw_eye_outline, detect_blinks
from app.model.detectors.face_expression_detector import expression_counts, detect_expression
from app.model.detectors.face_mesh_detector import FaceMeshDetector
from app.model.detectors.gaze_detector import extract_iris_center, extract_iris_landmarks, analyze_gaze_for_eye, \
    draw_iris_outline, draw_iris_center
from app.model.detectors.yawn_detector import extract_mouth_landmarks, calculate_mar, draw_mouth_state, detect_yawn

blink_counts = 0
blink_count_frames = 0

yawn_counts = 0
yaw_count_frames = 0

frame_count = 0
gaze_sample_rate = 15
gaze_left_count = 0
gaze_right_count = 0
gaze_center_count = 0
no_gaze = 0


face_detector = FaceMeshDetector()

class FacialCueAnalyzer:
    def __init__(self):
        self.facial_cues_data = {
            "blink_counts": 0,
            "yawn_counts": 0,
            "gaze_direction_counts": {
                "left": 0,
                "right": 0,
                "center": 0,
                "no_gaze":0
            },
            "face_expression_counts": {
                "happy": 0,
                "sad": 0,
                "angry": 0,
                "surprise": 0,
                "neutral": 0,
                "disgust": 0,
                "fear": 0,
                "no_face": 0
            }
        }
        self.running = False
        self.facial_cue_snap_shot_and_reset()
        self.reset_data()


    def start_facial_cue_detector(self):
        self.running = True
        capture = cv2.VideoCapture(0)

        while self.running:
            ret, frame = capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame, face_landmarks = face_detector.detect_face_landmarks(frame)
            if face_landmarks:
                for face_landmark in face_landmarks:
                    image_height, image_width, _ = frame.shape

                    # Detecting blinks
                    right_eye_points, left_eye_points = extract_eye_landmarks(face_landmark, image_width, image_height)
                    right_eye_ear = calculate_ear(right_eye_points)
                    left_eye_ear = calculate_ear(left_eye_points)

                    draw_eye_outline(frame, right_eye_points)
                    draw_eye_outline(frame, left_eye_points)

                    global blink_count_frames, blink_counts
                    blink_detected, blink_count_frames, blink_counts = detect_blinks(
                        right_eye_ear, left_eye_ear, blink_count_frames, blink_counts
                    )

                    if blink_detected:
                        print("Blink detected")
                        self.facial_cues_data["blink_counts"] += 1

                    cv2.putText(frame, f"Blink Detected: {blink_detected}",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255) if blink_detected else (0, 255, 0), 2)

                    # Detecting yawning
                    mouth_points = extract_mouth_landmarks(face_landmark, image_width, image_height)
                    mar = calculate_mar(mouth_points)
                    draw_mouth_state(frame, mouth_points, mar)

                    global yaw_count_frames, yawn_counts
                    yaw_count_frames, yawn_counts, yawn_detected = detect_yawn(mar, yaw_count_frames, yawn_counts)

                    if yawn_detected:
                        print("Yawn detected")
                        self.facial_cues_data["yawn_counts"] += 1

                    cv2.putText(frame, f"Yawn Detected: {yawn_detected}",
                                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255) if yawn_detected else (0, 255, 0), 2)

                    # Detecting gaze directions
                    left_iris_points, right_iris_points =  extract_iris_landmarks(face_landmark, image_width, image_height)
                    left_center, right_center = extract_iris_center(face_landmark, image_width, image_height)

                    left_gaze = analyze_gaze_for_eye(face_landmark, image_width, image_height, is_left_eye=True)
                    right_gaze = analyze_gaze_for_eye(face_landmark, image_width, image_height, is_left_eye=False)

                    gaze_direction = left_gaze

                    gaze_color = (0, 0, 255) if gaze_direction in ['Left', 'Right'] else (0, 255,0)
                    cv2.putText(frame, f"Gaze: {gaze_direction}",(10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,gaze_color, 2)

                    global frame_count, gaze_sample_rate, gaze_left_count, gaze_right_count, gaze_center_count, no_gaze
                    if frame_count % gaze_sample_rate == 0:
                        if gaze_direction == 'Left':
                            gaze_left_count += 1
                            self.facial_cues_data["gaze_direction_counts"]["left"] += 1
                        elif gaze_direction == 'Right':
                            gaze_right_count += 1
                            self.facial_cues_data["gaze_direction_counts"]["right"] += 1
                        elif gaze_direction == 'Center':
                            gaze_center_count += 1
                            self.facial_cues_data["gaze_direction_counts"]["center"] += 1
                        else:
                            no_gaze += 1
                            self.facial_cues_data["gaze_direction_counts"]["no_gaze"] += 1

                    frame_count += 1

                    draw_iris_outline(frame, right_iris_points)
                    draw_iris_outline(frame, left_iris_points)

                    draw_iris_center(frame, right_center)
                    draw_iris_center(frame, left_center)

                    # Detecting facial expressions
                    expression_counts, last_expression = detect_expression(frame)

                    if last_expression == "neutral":
                        expression_color = (0, 255, 0)  # Green
                    elif last_expression == "no_face":
                        expression_color = (0, 0, 255)  # Red
                    else:
                        expression_color = (255, 0, 0)  # Blue

                    cv2.putText(frame, f"Expression: {last_expression}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, expression_color, 2)

                    if last_expression in self.facial_cues_data["face_expression_counts"]:
                        self.facial_cues_data["face_expression_counts"][last_expression] += 1

            cv2.imshow("Facial Cue Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_facial_cue_detector()
                break

        capture.release()
        cv2.destroyAllWindows()
        face_detector.close()

    def facial_cue_snap_shot_and_reset(self):
        facial_cue_snap = self.facial_cues_data.copy()
        self.reset_data()
        return facial_cue_snap

    def reset_data(self):
        self.facial_cues_data = {
            "blink_counts": 0,
            "yawn_counts": 0,
            "gaze_direction_counts": {
                "left": 0,
                "right": 0,
                "center": 0,
                "no_gaze": 0
            },
            "face_expression_counts": {
                "happy": 0,
                "sad": 0,
                "angry": 0,
                "surprise": 0,
                "neutral": 0,
                "disgust": 0,
                "fear": 0,
                "no_face": 0
            }
        }

    def stop_facial_cue_detector(self):
        self.running = False
        self.reset_data()
        cv2.destroyAllWindows()
        face_detector.close()




