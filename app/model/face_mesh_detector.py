import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, max_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect_face_landmarks(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        face_landmarks = []
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                            circle_radius=1)
                    )
                face_landmarks.append(face.landmark)
        return frame, face_landmarks