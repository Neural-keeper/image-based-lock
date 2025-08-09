#has to be organized into classes and functions

import cv2

#-- This section takes care of the input collection --#


#-- This section contains the algorithm that analyzes the submitted picture --#
imagePath = 'input_image.jpg'

img = cv2.imread(imagePath)
# print(img.shape)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

faces = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.25, minNeighbors=4,
)

num_faces = len(faces)

print(num_faces)