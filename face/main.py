import mediapipe as mp

from face.Visualization import detector
from face.views import visualize

image = mp.Image.create_from_file(IMAGE_FILE)
detection_result = detector.detect(image)

visualized_image, num_faces_detected = visualize(image, detection_result)
