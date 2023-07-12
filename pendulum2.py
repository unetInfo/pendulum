import cv2 as cv
import numpy as np
from threading import Thread
from pygame import mixer

# Constants
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
RIGHT_CENTER_X_THRESHOLD = (VIDEO_WIDTH / 2) + 50  # Adjust this value based on your video frame width
LEFT_CENTER_X_ThRESHOLD = (VIDEO_WIDTH / 2) - 50
DIST_THRESHOLD = 60  # Threshold for distance comparison


videoCapture = cv.VideoCapture(0)
# Reduce the resolution
videoCapture.set(cv.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
videoCapture.set(cv.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
prevCircle = None


mixer.init(channels=64)
sound = mixer.Sound('Piano_C3.wav')
channel_number = 0

def play_sound():
    global channel_number
    mixer.Channel(channel_number).play(sound)
    channel_number = (channel_number + 1) % mixer.get_num_channels()
    
while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, minDist=100, param1=50, param2=30, minRadius=15, maxRadius=55)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            else:
                dist = np.linalg.norm(chosen[:2] - i[:2])
                if dist <= DIST_THRESHOLD:
                    chosen = i

        center_x = chosen[0]
        radius = chosen[2]

        # Draw circle and center point on the frame
        cv.circle(frame, (chosen[0], chosen[1]), radius, (0, 100, 100), 3)
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)

        # Check if the object crosses the center x-axis
        if center_x < RIGHT_CENTER_X_THRESHOLD and center_x > LEFT_CENTER_X_ThRESHOLD:
            play_sound()

    cv.imshow("circles", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()