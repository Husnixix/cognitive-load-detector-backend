import math
import cv2

ear_threshold = 0.21
eye_consec_frames = 3

def extract_eye_landmarks(face_landmarks, image_width, image_height):
    right_eye_landmarks = [33, 159, 158, 133, 153, 144]
    left_eye_landmarks = [263, 386, 385, 362, 380, 373]

    right_eye_points = []
    left_eye_points = []

    for index in right_eye_landmarks:
        landmark = face_landmarks[index]
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        right_eye_points.append((x, y))

    for index in left_eye_landmarks:
        landmark = face_landmarks[index]
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        left_eye_points.append((x, y))

    return right_eye_points, left_eye_points

def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points

    def distance(point_a, point_b):
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    vertical_a = distance(p2, p6)
    vertical_b = distance(p3, p5)
    horizontal = distance(p1, p4)

    ear = (vertical_a + vertical_b) / (2.0  * horizontal)
    return ear

def detect_blinks(right_eye_ear, left_eye_ear, blink_count_frames, blink_counts):
    blink_detected = False
    average_ear = (right_eye_ear + left_eye_ear) / 2.0
    if average_ear < ear_threshold:
        blink_count_frames += 1
    else:
        if blink_count_frames >= eye_consec_frames:
            blink_counts += 1
            blink_detected = True
        blink_count_frames = 0

    return blink_detected, blink_count_frames, blink_counts

def draw_eye_outline(frame, eye_points, color=(0, 0, 255)):
    number_of_points = len(eye_points)
    for i in range(number_of_points):
        start = eye_points[i]
        end = eye_points[(i + 1) % number_of_points]
        cv2.line(frame, start, end, color, 1)
