import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

camera = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

sensivity = 25
textList = ["Face Depth Reader", "Press 'q' to quit"]

while True:
    success, img = camera.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3

        f = 840
        d = (W * f) / w

        cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

        for i, text in enumerate(textList):
            singleHeight = 20 + int((int(d/sensivity)*sensivity)/4)
            scale = 0.4 + (int(d/sensivity)*sensivity)/75
            cv2.putText(imgText, text, (50, 50 + (i * singleHeight)),
                        cv2.FONT_ITALIC, scale, (255, 255, 255), 2)

    imgStacked = cvzone.stackImages([img, imgText], 2, 1)

    cv2.imshow("Face Depth Reader", imgStacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
