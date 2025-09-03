import cv2
from app.domain.dependencies.face_mesh_detector import FaceMeshDetector
from app.domain.dependencies.blink_detector import extract_eye_landmarks, calculate_ear, draw_eye_outline, detect_blinks
from app.domain.dependencies.face_expression_detector import detect_expression
from app.domain.dependencies.gaze_detector import (extract_iris_center, extract_iris_landmarks, analyze_gaze_for_eye,
                                                   draw_iris_outline, draw_iris_center)
from app.domain.dependencies.yawn_detector import extract_mouth_landmarks, calculate_mar, draw_mouth_state, detect_yawn


TEXT_POSITIONS = {
    "blink": (10, 30),
    "yawn": (10, 60),
    "gaze": (10, 90),
    "expression": (10, 120)
}

COLORS = {
    "ok": (0, 255, 0),
    "alert": (0, 0, 255),
    "info": (255, 0, 0)
}


class FacialCueDetector:
    def __init__(self):
        self.face_detector = FaceMeshDetector()
        self.gaze_sample_rate = 15
        self.frame_count = 0
        self.running = False
        self.reset_data()

    def _init_camera(self):
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not capture.isOpened():
            print("[ERROR] Camera index 0 failed, trying index 1...")
            capture.release()
            capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not capture.isOpened():
            print("[FATAL] No camera found. Check permissions.")
            return None
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return capture

    def _cleanup(self, capture):
        capture.release()
        cv2.destroyAllWindows()
        self.face_detector.close()

    # ================= MAIN LOOP =================
    def start_facial_cue_detector(self):
        self.running = True
        capture = self._init_camera()
        if not capture:
            return

        while self.running:
            ret, frame = capture.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            frame = self._process_frame(frame)
            cv2.imshow("Facial Cue Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_facial_cue_detector()
                break

        self._cleanup(capture)

    def _process_frame(self, frame):
        frame, landmarks = self.face_detector.detect_face_landmarks(frame)
        if landmarks:
            for lm in landmarks:
                h, w, _ = frame.shape
                frame = self._process_blinks(frame, lm, w, h)
                frame = self._process_yawn(frame, lm, w, h)
                frame = self._process_gaze(frame, lm, w, h)
                frame = self._process_expression(frame)
        return frame

    # ================= DETECTION MODULES =================
    def _process_blinks(self, frame, lm, w, h):
        right_eye, left_eye = extract_eye_landmarks(lm, w, h)
        right_ear, left_ear = calculate_ear(right_eye), calculate_ear(left_eye)
        draw_eye_outline(frame, right_eye)
        draw_eye_outline(frame, left_eye)

        blink_detected, _, _ = detect_blinks(right_ear, left_ear, 0, 0)
        if blink_detected:
            self.facial_cues_data["blink_counts"] += 1

        cv2.putText(frame, f"Blink Detected: {blink_detected}",
                    TEXT_POSITIONS["blink"], cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLORS["alert"] if blink_detected else COLORS["ok"], 2)
        return frame

    def _process_yawn(self, frame, lm, w, h):
        mouth_points = extract_mouth_landmarks(lm, w, h)
        mar = calculate_mar(mouth_points)
        draw_mouth_state(frame, mouth_points, mar)

        _, _, yawn_detected = detect_yawn(mar, 0, 0)
        if yawn_detected:
            self.facial_cues_data["yawn_counts"] += 1

        cv2.putText(frame, f"Yawn Detected: {yawn_detected}",
                    TEXT_POSITIONS["yawn"], cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLORS["alert"] if yawn_detected else COLORS["ok"], 2)
        return frame

    def _process_gaze(self, frame, lm, w, h):
        left_iris, right_iris = extract_iris_landmarks(lm, w, h)
        left_center, right_center = extract_iris_center(lm, w, h)

        gaze_direction = analyze_gaze_for_eye(lm, w, h, is_left_eye=True)
        cv2.putText(frame, f"Gaze: {gaze_direction}",
                    TEXT_POSITIONS["gaze"], cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLORS["alert"] if gaze_direction in ["Left", "Right"] else COLORS["ok"], 2)

        if self.frame_count % self.gaze_sample_rate == 0:
            self.facial_cues_data["gaze_direction_counts"][gaze_direction.lower() if gaze_direction else "no_gaze"] += 1
        self.frame_count += 1

        draw_iris_outline(frame, left_iris)
        draw_iris_outline(frame, right_iris)
        draw_iris_center(frame, left_center)
        draw_iris_center(frame, right_center)
        return frame

    def _process_expression(self, frame):
        _, last_expression = detect_expression(frame)
        color = COLORS["ok"] if last_expression == "neutral" else (
            COLORS["alert"] if last_expression == "no_face" else COLORS["info"]
        )
        cv2.putText(frame, f"Expression: {last_expression}",
                    TEXT_POSITIONS["expression"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if last_expression in self.facial_cues_data["face_expression_counts"]:
            self.facial_cues_data["face_expression_counts"][last_expression] += 1
        return frame

    # ================= STATE MANAGEMENT =================
    def reset_data(self):
        self.facial_cues_data = {
            "blink_counts": 0,
            "yawn_counts": 0,
            "gaze_direction_counts": {
                "left": 0, "right": 0, "center": 0, "no_gaze": 0
            },
            "face_expression_counts": {
                "happy": 0, "sad": 0, "angry": 0, "surprise": 0,
                "neutral": 0, "disgust": 0, "fear": 0, "no_face": 0
            }
        }

    def facial_cue_snap_shot_and_reset(self):
        snapshot = self.facial_cues_data.copy()
        self.reset_data()
        return snapshot

    def stop_facial_cue_detector(self):
        self.running = False
        self.reset_data()
        cv2.destroyAllWindows()
        self.face_detector.close()
