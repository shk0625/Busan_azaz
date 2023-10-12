import cv2
import mediapipe as mp

from Visualization import detector
from views import visualize

IMAGE_FILE = 'image.jpg'

image = mp.Image.create_from_file(IMAGE_FILE)
detection_result = detector.detect(image)

visualized_image, num_faces_detected = visualize(image, detection_result)

img = cv2.imread(IMAGE_FILE)
cv2.imshow(img)