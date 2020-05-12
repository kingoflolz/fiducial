import time
from cv2 import *
# initialize the camera
cam = VideoCapture(0)# 0 -> index of camera

for i in range(10):
    s, img = cam.read()

i = 0
while True:
    s, img = cam.read()
    if s:    # frame captured without any errors
        i += 1
        if i % 60 == 0:
            print("saved")
            imwrite(f"{i}.jpg",img) #save image

    if i > 1200:
        break