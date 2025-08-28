import math
import cv2

yawn_threshold = 0.5
yawn_consec_frames = 12

def extract_mouth_landmarks(face_landmarks, image_width, image_height):
    mouth_indices = [13, 14, 81, 178, 78, 308]
    mouth_points = []

    for index in mouth_indices:
        landmark = face_landmarks[index]
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        mouth_points.append((x, y))

    return mouth_points

def calculate_mar(mouth_points):
    upper_outer = mouth_points[0]
    lower_outer = mouth_points[1]
    upper_inner = mouth_points[2]
    lower_inner = mouth_points[3]
    left_corner = mouth_points[4]
    right_corner = mouth_points[5]

    def distance(point_a, point_b):
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    vertical_a = distance(upper_outer, lower_outer)
    vertical_b = distance(upper_inner, lower_inner)
    horizontal = distance(left_corner, right_corner)

    mar = (vertical_a + vertical_b) / (2.0 * horizontal)
    return mar

def detect_yawn(mar, yawn_counter_frames, yawn_counts):
    yawn_detected = False

    if mar > yawn_threshold:
        yawn_counter_frames += 1
    else:
        if yawn_counter_frames >= yawn_consec_frames:
            yawn_counts += 1
            yawn_detected = True
        yawn_counter_frames = 0

    return yawn_counter_frames, yawn_counts, yawn_detected

def draw_mouth_state(frame, mouth_points, mar):
    left_corner = mouth_points[4]
    right_corner = mouth_points[5]
    upper_point = mouth_points[0]
    lower_point = mouth_points[1]

    color = (0, 0, 255) if mar > yawn_threshold else (0, 255, 0)
    cv2.line(frame, left_corner, right_corner, color, 2)
    cv2.line(frame, upper_point, lower_point, color, 2)

