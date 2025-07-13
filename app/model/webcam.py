import cv2

class Webcam:
    def __init__(self, camera_index=0):
        self.capture = cv2.VideoCapture(camera_index)

    def read_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            print("Failed to capture frame")
            return None
        return cv2.flip(frame, 1)

    def release_frame(self):
        self.capture.release()
        cv2.destroyAllWindows()