import math
import cv2

EAR_THRESHOLD = 0.21
EYE_CONSEC_FRAMES = 3

def extract_eye_landmarks(landmarks, image_width, image_height):
    right_eye = [33, 159, 158, 133, 153, 144]
    left_eye = [263, 386, 385, 362, 380, 373]

    right_eye_points = []
    left_eye_points = []

    for index in right_eye:
        landmark = landmarks[index]
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        right_eye_points.append((x, y))

    for index in left_eye:
        landmark = landmarks[index]
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        left_eye_points.append((x, y))

    return right_eye_points, left_eye_points

def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points

    def distance(point_a, point_b):
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    EAR = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return EAR

def draw_eye_outline(frame, eye_points, color=(0, 0, 255)):
    num_points = len(eye_points)
    for i in range(num_points):
        start_point = eye_points[i]
        end_point = eye_points[(i + 1) % num_points]
        cv2.line(frame, start_point, end_point, color, 1)

def detect_blinks(right_eye_ear, left_eye_ear, blink_count_frames, blink_counts):
    blink_detected = False
    average_ear = (right_eye_ear + left_eye_ear) / 2.0

    if average_ear < EAR_THRESHOLD:
        blink_count_frames += 1
    else:
        if blink_count_frames >= EYE_CONSEC_FRAMES:
            blink_counts += 1
            blink_detected = True
        blink_count_frames = 0

    return blink_count_frames, blink_counts, blink_detected
