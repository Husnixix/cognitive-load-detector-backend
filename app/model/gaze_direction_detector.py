import numpy as np
import cv2


def extract_iris_landmarks(face_landmark, image_width, image_height):
    # FIXED: Correct MediaPipe iris landmark indices for 468-point model
    # Left iris (from subject's perspective - appears on right in camera)
    left_iris_ids = [474, 475, 476, 477] if len(face_landmark) > 468 else [468, 469, 470, 471]
    # Right iris (from subject's perspective - appears on left in camera)
    right_iris_ids = [469, 470, 471, 472] if len(face_landmark) > 468 else [473, 474, 475, 476]

    right_iris_points = []
    left_iris_points = []

    # Use try-catch to handle index errors
    try:
        for idx in right_iris_ids:
            landmark = face_landmark[idx]
            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
            right_iris_points.append((x, y))

        for idx in left_iris_ids:
            landmark = face_landmark[idx]
            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
            left_iris_points.append((x, y))
    except IndexError:
        # Fallback: use approximate iris positions from pupil center
        return get_iris_fallback(face_landmark, image_width, image_height)

    return right_iris_points, left_iris_points


def extract_iris_center(face_landmark, image_width, image_height):
    # FIXED: Use consistent indices and add fallback
    try:
        if len(face_landmark) > 468:
            left_iris = face_landmark[468]  # Left iris center
            right_iris = face_landmark[473]  # Right iris center
        else:
            # Fallback: estimate from eye center
            left_iris = face_landmark[468] if len(face_landmark) > 468 else face_landmark[159]
            right_iris = face_landmark[473] if len(face_landmark) > 468 else face_landmark[145]
    except IndexError:
        # Ultimate fallback
        left_iris = face_landmark[159]  # Left eye center approximation
        right_iris = face_landmark[145]  # Right eye center approximation

    left_center = (int(left_iris.x * image_width), int(left_iris.y * image_height))
    right_center = (int(right_iris.x * image_width), int(right_iris.y * image_height))

    return left_center, right_center


def get_iris_fallback(face_landmark, image_width, image_height):
    """Fallback method when iris landmarks aren't available"""
    # Use eye center points as approximation
    left_eye_center = face_landmark[159]
    right_eye_center = face_landmark[145]

    left_center = (int(left_eye_center.x * image_width), int(left_eye_center.y * image_height))
    right_center = (int(right_eye_center.x * image_width), int(right_eye_center.y * image_height))

    # Create small circles around centers as "iris points"
    radius = 3
    left_iris_points = [
        (left_center[0] - radius, left_center[1]),
        (left_center[0], left_center[1] - radius),
        (left_center[0] + radius, left_center[1]),
        (left_center[0], left_center[1] + radius)
    ]
    right_iris_points = [
        (right_center[0] - radius, right_center[1]),
        (right_center[0], right_center[1] - radius),
        (right_center[0] + radius, right_center[1]),
        (right_center[0], right_center[1] + radius)
    ]

    return right_iris_points, left_iris_points


def get_eye_corners(face_landmark, image_width, image_height, is_left_eye=True):
    """ADDED: Get actual eye corner landmarks instead of using iris points"""
    if is_left_eye:
        # Left eye corners (from subject's perspective)
        inner_corner_idx = 133  # Inner corner of left eye
        outer_corner_idx = 33  # Outer corner of left eye
    else:
        # Right eye corners (from subject's perspective)
        inner_corner_idx = 362  # Inner corner of right eye
        outer_corner_idx = 263  # Outer corner of right eye

    inner_corner = face_landmark[inner_corner_idx]
    outer_corner = face_landmark[outer_corner_idx]

    inner_point = (int(inner_corner.x * image_width), int(inner_corner.y * image_height))
    outer_point = (int(outer_corner.x * image_width), int(outer_corner.y * image_height))

    return inner_point, outer_point


def draw_iris_outline(frame, iris_points, color=(0, 255, 0), thickness=2):
    if len(iris_points) >= 3:  # Need at least 3 points for polygon
        points = np.array(iris_points, dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)


def draw_iris_center(frame, center_point, color=(0, 0, 255), radius=3):
    cv2.circle(frame, center_point, radius, color, -1)


def detect_gaze_direction(eye_points, iris_position, face_landmark=None, image_width=None, image_height=None,
                          is_left_eye=True):
    """
    COMPLETELY REWRITTEN: Now uses proper eye corners instead of iris points
    """
    # If we have face_landmark data, use proper eye corners
    if face_landmark is not None and image_width is not None and image_height is not None:
        inner_corner, outer_corner = get_eye_corners(face_landmark, image_width, image_height, is_left_eye)

        # Calculate eye width
        eye_width = abs(outer_corner[0] - inner_corner[0])

        # Calculate iris position relative to inner corner
        iris_offset = abs(iris_position[0] - inner_corner[0])

        if eye_width == 0:
            return 'Center'

        ratio = iris_offset / eye_width

        # FIXED: Adjusted thresholds and logic
        if ratio < 0.35:
            return 'Left' if is_left_eye else 'Right'  # Looking towards inner corner
        elif ratio > 0.65:
            return 'Right' if is_left_eye else 'Left'  # Looking towards outer corner
        else:
            return 'Center'

    else:
        # Fallback to original method (but fixed)
        if not eye_points or len(eye_points) == 0:
            return 'Center'

        eye_left_corner = min([p[0] for p in eye_points])
        eye_right_corner = max([p[0] for p in eye_points])

        eye_width = eye_right_corner - eye_left_corner

        if eye_width == 0:
            return 'Center'

        iris_offset = iris_position[0] - eye_left_corner
        ratio = iris_offset / eye_width

        # FIXED: Corrected thresholds
        if ratio <= 0.35:
            return 'Left'
        elif ratio >= 0.65:
            return 'Right'
        else:
            return 'Center'


# ADDED: Utility function to determine which eye we're analyzing
def analyze_gaze_for_eye(face_landmark, image_width, image_height, is_left_eye=True):
    """
    Complete gaze analysis for one eye
    """
    iris_points_right, iris_points_left = extract_iris_landmarks(face_landmark, image_width, image_height)
    left_center, right_center = extract_iris_center(face_landmark, image_width, image_height)

    if is_left_eye:
        return detect_gaze_direction(
            iris_points_left, left_center, face_landmark,
            image_width, image_height, is_left_eye=True
        )
    else:
        return detect_gaze_direction(
            iris_points_right, right_center, face_landmark,
            image_width, image_height, is_left_eye=False
        )