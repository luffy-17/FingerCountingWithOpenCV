import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)

out = cv2.VideoWriter("handTrackinAndCounting2.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20,(640,480))

cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
folder_path = "FingerImages"

my_list = os.listdir(folder_path)

overlay_list = []

for imPath in my_list:
    image = cv2.imread(f'{folder_path}/{imPath}')
    # Resized images because images were very big and downloaded from google
    image1 = cv2.resize(image, (150,200))
    overlay_list.append(image1)

# This module created to detect the human palm with hand
# landmarks model from MediaPipe (21 total landmarks starting from zero)

detector = htm.handDetector(detectionCon=0.75)

# tips of the each fingers starting from thumb
# [thumb, index, middle, ring, pinky]
tipIds = [4,8,12,16,20]

while True:
    success, img = cap.read()
    img, label = detector.findHands(img)
    lm_list = detector.findPosition(img, draw=False)

    if len(lm_list) != 0:
        fingers =[]
        # left hand Thumb
        if label == 'Right':
            if lm_list[tipIds[0]][1] < lm_list[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif label == 'Left':
            # Thumb
            if lm_list[tipIds[0]][1] > lm_list[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # other four finger
        for id in range(1, 5):
            if lm_list[tipIds[id]][2] < lm_list[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

        h, w, c = overlay_list[total_fingers - 1].shape
        img[0:h, 0:w] = overlay_list[total_fingers-1]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,0), 3)
    cv2.imshow("Image", img)
    out.write(img)
    cv2.waitKey(1)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()