from deepface import DeepFace
import cv2

expression_counts = {
    "happy": 0,
    "sad": 0,
    "angry": 0,
    "surprise": 0,
    "neutral": 0,
    "disgust": 0,
    "fear": 0,
    "no_face": 0
}

last_expression = "neutral"

def detect_expression(frame):
    global last_expression

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        emotion = result['dominant_emotion']

        if emotion in expression_counts:
            expression_counts[emotion] += 1
            last_expression = emotion

    except Exception as e:
        print(f"[ExpressionAnalyzer] Error: {e}")
        expression_counts["no_face"] += 1
        last_expression = "no_face"

    return expression_counts, last_expression
