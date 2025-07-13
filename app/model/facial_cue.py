import cv2

from app.model.eye_blink_detector import calculate_ear, extract_eye_landmarks, draw_eye_outline, detect_blinks
from app.model.face_expression_detector import detect_expression
from app.model.gaze_direction_detector import extract_iris_landmarks, draw_iris_outline, extract_iris_center, \
    draw_iris_center, detect_gaze_direction, analyze_gaze_for_eye  # ADDED: analyze_gaze_for_eye
from app.model.webcam import Webcam
from app.model.face_mesh_detector import FaceMeshDetector
from app.model.yawning_detector import extract_mouth_landmarks, calculate_mar, detect_yawn

webcam = Webcam()
detector = FaceMeshDetector()



blink_count_frames = 0
blink_counts = 0

yawn_counter_frames = 0
yawn_counts = 0

# ADDED: Frame counting to avoid counting every frame
frame_counter = 0
gaze_sample_rate = 15  # Sample every 15 frames (every 0.5 seconds at 30fps)

gaze_left_count = 0
gaze_right_count = 0
gaze_center_count = 0


session_metrics = {
    "blink_count": 0,
    "yawn_count": 0,
    "gaze_direction_count": {
        "left": 0,
        "right": 0,
        "center": 0,
    },
    "face_expression_count": {
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


while True:
    frame = webcam.read_frame()
    if frame is None:
        break

    frame, face_landmarks = detector.detect_face_landmarks(frame, draw=True)
    frame_counter += 1  # ADDED: Increment frame counter

    if face_landmarks:
        for face_landmark in face_landmarks:
            image_height, image_width, _ = frame.shape
            right_eye_points, left_eye_points = extract_eye_landmarks(face_landmark, image_width, image_height)

            right_eye_ear = calculate_ear(right_eye_points)
            left_eye_ear = calculate_ear(left_eye_points)

            # Draw eye outlines
            draw_eye_outline(frame, right_eye_points)
            draw_eye_outline(frame, left_eye_points)

            # Blink detection
            blink_count_frames, blink_counts, blink_detected = detect_blinks(
                right_eye_ear, left_eye_ear, blink_count_frames, blink_counts
            )

            if blink_detected:
                print(f"Blink detected! Total: {blink_counts}")
                session_metrics["blink_count"] = blink_counts

            mouth_points = extract_mouth_landmarks(face_landmark, image_width, image_height)
            mar = calculate_mar(mouth_points)
            draw_eye_outline(frame, mouth_points, color=(255, 0, 0))

            yawn_counter_frames, yawn_counts, yawn_detected = detect_yawn(mar, yawn_counter_frames, yawn_counts)

            if yawn_detected:
                print(f"Yawn detected! Total: {yawn_counts}")
                session_metrics["yawn_count"] = yawn_counts

            # FIXED: Gaze detection - extract iris data
            right_iris_points, left_iris_points = extract_iris_landmarks(face_landmark, image_width, image_height)
            left_center, right_center = extract_iris_center(face_landmark, image_width, image_height)

            draw_iris_outline(frame, right_iris_points)
            draw_iris_outline(frame, left_iris_points)

            draw_iris_center(frame, right_center)
            draw_iris_center(frame, left_center)

            # FIXED: Use improved gaze detection with face_landmark data
            left_gaze = analyze_gaze_for_eye(face_landmark, image_width, image_height, is_left_eye=True)
            right_gaze = analyze_gaze_for_eye(face_landmark, image_width, image_height, is_left_eye=False)

            # You can average or just pick one eye to count, here we pick left eye for example:
            gaze_direction = left_gaze

            # FIXED: Only count every N frames to avoid rapid counting
            if frame_counter % gaze_sample_rate == 0:
                if gaze_direction == 'Left':
                    gaze_left_count += 1
                elif gaze_direction == 'Right':
                    gaze_right_count += 1
                else:
                    gaze_center_count += 1

                print(
                    f"Gaze: {gaze_direction} | Left: {gaze_left_count} | Right: {gaze_right_count} | Center: {gaze_center_count}")


                cv2.putText(frame, f"Gaze: {gaze_direction}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                session_metrics["gaze_direction_count"]["left"] = gaze_left_count
                session_metrics["gaze_direction_count"]["right"] = gaze_right_count
                session_metrics["gaze_direction_count"]["center"] = gaze_center_count
                expression_counts, last_expression = detect_expression(frame)
                print(f'Expression: {last_expression} | Counts: {expression_counts}')
                for emotion in expression_counts:
                    session_metrics["face_expression_count"][emotion] = expression_counts[emotion]

            # ADDED: Display counts on frame for debugging
            cv2.putText(frame, f"L:{gaze_left_count} R:{gaze_right_count} C:{gaze_center_count}",
                        (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(f"Blinks: {session_metrics['blink_count']}")
    print(f"Yawns: {session_metrics['yawn_count']}")
    print(f"Gaze counts: {session_metrics['gaze_direction_count']}")
    print(f"Expression counts: {session_metrics['face_expression_count']}")

webcam.release_frame()
