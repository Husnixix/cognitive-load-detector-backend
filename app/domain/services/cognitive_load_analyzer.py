# Moved from app.model.cognitive_load_analyzer to domain services as pure domain logic

thresholds = {
    "blink_count": (10, 15, 20),
    "yawn_count": (1, 2, 3),
    "gaze_center_ratio": (0.10, 0.15, 0.20),
    "negative_expression_ratio": (0.20, 0.40, 0.60),
    "typing_speed": (2, 4, 6),
    "error_rate": (2, 5, 10),
    "pause_rate": (0.5, 1.0, 2.0)
}

weights = {
    "blinking": 0.30,
    "yawning": 0.30,
    "gaze": 0.20,
    "expression": 0.15,
    "keystroke": 0.05,
}


class CognitiveLoadAnalyzer:
    def __init__(self):
        self.cognitive_load = {
            "score": 0,
            "label": ["Low", "Medium", "High"]
        }

    def score_feature(self, facial_cue_data, keystroke_data):
        # Blinking
        low, medium, high = thresholds["blink_count"]
        if facial_cue_data["blink_counts"] <= low:
            self.cognitive_load["score"] += 0
        elif facial_cue_data["blink_counts"] <= medium:
            self.cognitive_load["score"] += 50
        elif facial_cue_data["blink_counts"] <= high:
            self.cognitive_load["score"] += 80
        else:
            self.cognitive_load["score"] += 100

        blink_score = self.cognitive_load["score"] * weights["blinking"]
        self.cognitive_load["score"] = 0

        # Yawning
        low, medium, high = thresholds["yawn_count"]
        if facial_cue_data["yawn_counts"] <= low:
            self.cognitive_load["score"] += 0
        elif facial_cue_data["yawn_counts"] <= medium:
            self.cognitive_load["score"] += 50
        elif facial_cue_data["yawn_counts"] <= high:
            self.cognitive_load["score"] += 80
        else:
            self.cognitive_load["score"] += 100

        yawn_score = self.cognitive_load["score"] * weights["yawning"]
        self.cognitive_load["score"] = 0

        # Gaze directions
        total_gaze_counts = sum(facial_cue_data["gaze_direction_counts"].values())
        low, medium, high = thresholds["gaze_center_ratio"]
        if total_gaze_counts > 0:
            gaze_center_ratio = facial_cue_data["gaze_direction_counts"]["center"] / total_gaze_counts
        else:
            gaze_center_ratio = 0

        if gaze_center_ratio >= high:
            self.cognitive_load["score"] += 0
        elif gaze_center_ratio >= medium:
            self.cognitive_load["score"] += 20
        elif gaze_center_ratio >= low:
            self.cognitive_load["score"] += 50
        else:
            self.cognitive_load["score"] += 100

        gaze_score = self.cognitive_load["score"] * weights["gaze"]
        self.cognitive_load["score"] = 0

        # Facial expressions
        face_counts = facial_cue_data["face_expression_counts"]
        negative_expression_count = (
                face_counts["angry"] +
                face_counts["sad"] +
                face_counts["disgust"] +
                face_counts["fear"]
        )

        total_expression_counts = sum(face_counts.values()) - face_counts.get("no_face", 0)

        if total_expression_counts > 0:
            negative_expression_ratio = negative_expression_count / total_expression_counts
        else:
            negative_expression_ratio = 0

        low, medium, high = thresholds["negative_expression_ratio"]
        if negative_expression_ratio <= low:
            self.cognitive_load["score"] += 0
        elif negative_expression_ratio <= medium:
            self.cognitive_load["score"] += 50
        elif negative_expression_ratio <= high:
            self.cognitive_load["score"] += 80
        else:
            self.cognitive_load["score"] += 100

        expression_score = self.cognitive_load["score"] * weights["expression"]
        self.cognitive_load["score"] = 0

        # Keystroke (typing speed)
        low, medium, high = thresholds["typing_speed"]
        ts = keystroke_data["typing_speed"]
        if ts >= high:
            self.cognitive_load["score"] += 0
        elif ts >= medium:
            self.cognitive_load["score"] += 20
        elif ts >= low:
            self.cognitive_load["score"] += 60
        else:
            self.cognitive_load["score"] += 100

        typing_speed_score = self.cognitive_load["score"]
        self.cognitive_load["score"] = 0

        # Keystroke (error rate)
        low, medium, high = thresholds["error_rate"]
        error_rate = keystroke_data["error_rate"]

        if error_rate <= low:
            self.cognitive_load["score"] += 0
        elif error_rate <= medium:
            self.cognitive_load["score"] += 20
        elif error_rate <= high:
            self.cognitive_load["score"] += 60
        else:
            self.cognitive_load["score"] += 100

        error_rate_score = self.cognitive_load["score"]
        self.cognitive_load["score"] = 0

        # Keystroke (pause rate)
        low, medium, high = thresholds["pause_rate"]
        pause_rate = keystroke_data["pause_rate"]

        if pause_rate <= low:
            self.cognitive_load["score"] += 0
        elif pause_rate <= medium:
            self.cognitive_load["score"] += 20
        elif pause_rate <= high:
            self.cognitive_load["score"] += 60
        else:
            self.cognitive_load["score"] += 100

        pause_score = self.cognitive_load["score"]
        self.cognitive_load["score"] = 0

        keystroke_data_scores = ((typing_speed_score + error_rate_score + pause_score)/3)
        keystroke_score = keystroke_data_scores * weights["keystroke"]

        self.cognitive_load["score"] = int(blink_score + yawn_score + gaze_score + expression_score + keystroke_score)
        return self.cognitive_load["score"]

    def get_score_and_label(self, score):
        if score <= 40:
            return score, self.cognitive_load["label"][0]
        elif score <= 70:
            return score, self.cognitive_load["label"][1]
        else:
            return score, self.cognitive_load["label"][2]
