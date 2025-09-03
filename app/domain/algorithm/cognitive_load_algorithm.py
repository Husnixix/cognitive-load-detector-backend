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


class CognitiveLoadAlgorithm:
    def __init__(self):
        self.labels = ["Low", "Medium", "High"]

    def _threshold_score(self, value, low, medium, high, scores=(0, 75, 100, 0)):
        """Generic threshold scoring helper."""
        if value <= low:
            return scores[0]
        elif value <= medium:
            return scores[1]
        elif value <= high:
            return scores[2]
        return scores[3]

    def _gaze_score(self, gaze_counts):
        total = sum(gaze_counts.values())
        ratio = gaze_counts.get("center", 0) / total if total else 0
        return self._threshold_score(ratio, *thresholds["gaze_center_ratio"])

    def _expression_score(self, face_counts):
        neg = face_counts.get("angry", 0) + face_counts.get("sad", 0) \
              + face_counts.get("disgust", 0) + face_counts.get("fear", 0)
        total = sum(face_counts.values()) - face_counts.get("no_face", 0)
        ratio = neg / total if total else 0
        return self._threshold_score(ratio, *thresholds["negative_expression_ratio"], scores=(0, 50, 100, 0))

    def _keystroke_score(self, data):
        ts = self._threshold_score(data["typing_speed"], *thresholds["typing_speed"], scores=(0, 50, 100, 0))
        er = self._threshold_score(data["error_rate"], *thresholds["error_rate"], scores=(0, 50, 100, 0))
        pr = self._threshold_score(data["pause_rate"], *thresholds["pause_rate"], scores=(0, 50, 100, 0))
        return (ts + er + pr) / 3

    def score_feature(self, facial_cue_data, keystroke_data):
        blink = self._threshold_score(facial_cue_data["blink_counts"], *thresholds["blink_count"]) * weights["blinking"]
        yawn = self._threshold_score(facial_cue_data["yawn_counts"], *thresholds["yawn_count"]) * weights["yawning"]
        gaze = self._gaze_score(facial_cue_data["gaze_direction_counts"]) * weights["gaze"]
        expr = self._expression_score(facial_cue_data["face_expression_counts"]) * weights["expression"]
        key = self._keystroke_score(keystroke_data) * weights["keystroke"]

        total_score = int(blink + yawn + gaze + expr + key)
        return total_score

    def get_score_and_label(self, score):
        if score <= 40:
            return score, self.labels[0]
        elif score <= 70:
            return score, self.labels[1]
        return score, self.labels[2]
