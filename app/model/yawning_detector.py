import math

YAWN_THRESHOLD = 0.5  # You may need to tune this value
YAWN_CONSEC_FRAMES =  15 # Number of frames mouth should stay open to confirm yawn


def extract_mouth_landmarks(landmarks, image_width, image_height):
    # Index numbers for mouth points
    mouth_indices = [13, 14, 81, 178, 78, 308]

    mouth_points = []
    for index in mouth_indices:
        landmark = landmarks[index]
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        mouth_points.append((x, y))

    return mouth_points

def calculate_mar(mouth_points):
    upper_outer = mouth_points[0]  # 13
    lower_outer = mouth_points[1]  # 14
    upper_inner = mouth_points[2]  # 81
    lower_inner = mouth_points[3]  # 178
    left_corner = mouth_points[4]  # 78
    right_corner = mouth_points[5] # 308

    def distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    vertical1 = distance(upper_outer, lower_outer)
    vertical2 = distance(upper_inner, lower_inner)
    horizontal = distance(left_corner, right_corner)

    mar = (vertical1 + vertical2) / (2.0 * horizontal)
    return mar

def detect_yawn(mar, yawn_counter_frames, yawn_counts):
    yawn_detected = False

    if mar > YAWN_THRESHOLD:
        yawn_counter_frames += 1  # Mouth is open → count frames
    else:
        if yawn_counter_frames >= YAWN_CONSEC_FRAMES:
            yawn_counts += 1  # Yawn confirmed
            yawn_detected = True  # Mark that a yawn happened
        yawn_counter_frames = 0  # Reset counter

    return yawn_counter_frames, yawn_counts, yawn_detected