import cv2 as cv
import numpy as np
import pygame
from pygame import mixer

mixer.init(channels=64)
sound = mixer.Sound('Piano_C3.wav')
channel_number = 0

def play_sound():
    global channel_number
    mixer.Channel(channel_number).play(sound)
    channel_number = (channel_number + 1) % mixer.get_num_channels()

# Constants
DIST_THRESHOLD = 100  # Threshold for distance comparison
RADIUS_THRESHOLD = 50  # Threshold for radius comparison
NUM_CIRCLES = 2  # Number of circles we're interested in

videoCapture = cv.VideoCapture(0)

# Reduce the resolution
videoCapture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
videoCapture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

prev_circles = [None] * NUM_CIRCLES
prev_dxs = [None] * NUM_CIRCLES


def find_best_matches(prev_circles, new_circles, num_circles):
    """Find the best matches in new_circles for each circle in prev_circles."""
    if all(circle is None for circle in prev_circles):  # If there were no previous circles, just use the new ones
        return new_circles[:num_circles]
    
    matches = []
    for new_circle in new_circles:
        best_match = min(prev_circles, key=lambda c: abs(c[2] - new_circle[2]) if c is not None else float('inf'))
        matches.append((abs(best_match[2] - new_circle[2]), new_circle, best_match))
    matches.sort(key=lambda x: x[0])  # The best matches are now at the front of the list

    # Only keep as many matches as we want circles
    return [match[1] for match in matches[:num_circles]]

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (11, 11), 0)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=200)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen_circles = find_best_matches(prev_circles, circles[0], NUM_CIRCLES)
        for i, chosen in enumerate(chosen_circles):
            center_x = chosen[0]
            radius = chosen[2]

            if prev_circles[i] is not None:
                dx = center_x - prev_circles[i][0]
                if prev_dxs[i] is not None:
                    if np.sign(dx) != np.sign(prev_dxs[i]):
                        play_sound()
                prev_dxs[i] = dx

            # Draw circle and center point on the frame
            cv.circle(frame, (chosen[0], chosen[1]), radius, (0, 100, 100), 3)
            cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)

            prev_circles[i] = chosen

    cv.imshow("circles", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv.destroyAllWindows()
