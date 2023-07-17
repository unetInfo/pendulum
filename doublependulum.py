from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
import time
from threading import Thread
from playsound import playsound

# Constants
RIGHT_CENTER_X_THRESHOLD = 280  # Adjust this value based on your video frame width
LEFT_CENTER_X_THRESHOLD = 320
DIST_THRESHOLD = 100  # Threshold for distance comparison

def play_sound():
    playsound('Piano_C3.mp3')
    
def orange():
        # Construct a mask for orange color, perform dilations and erosions to remove blobs
        mask = cv.inRange(hsv, orangeLower, orangeUpper)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)

        # Find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        orangeCenter = None

        # Proceed when a contour is found
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)  # Find the largest contour in the mask
            ((x, y), radius) = cv.minEnclosingCircle(c)  # Compute the minimum enclosing circle
            M = cv.moments(c)
            if M["m00"] != 0:
                orangeCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Centroid

                if radius > 10:
                    cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv.circle(frame, orangeCenter, 5, (0, 0, 255), -1)
        
        # pts.appendleft(orangeCenter)        
            
        # for i in range(1, len(pts)):
        #     if pts[i - 1] is None or pts[i] is None:
        #         continue

        #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #     cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness) 
        
        # return orangeCenter

def purple():
        mask = cv.inRange(hsv, purpleLower, purpleUpper)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)

        # Find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        purpleCenter = None

        # Proceed when a contour is found
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)  # Find the largest contour in the mask
            ((x, y), radius) = cv.minEnclosingCircle(c)  # Compute the minimum enclosing circle
            M = cv.moments(c)
            if M["m00"] != 0:
                purpleCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Centroid

                if radius > 10:
                    cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv.circle(frame, purpleCenter, 5, (0, 0, 255), -1)
                    
        # pts.appendleft(purpleCenter)        
            
        # for i in range(1, len(pts)):
        #     if pts[i - 1] is None or pts[i] is None:
        #         continue

        #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #     cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)            
        
        # return purpleCenter
        
def red():
        mask = cv.inRange(hsv, redLower, redUpper)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)

        # Find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        redCenter = None

        # Proceed when a contour is found
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)  # Find the largest contour in the mask
            ((x, y), radius) = cv.minEnclosingCircle(c)  # Compute the minimum enclosing circle
            M = cv.moments(c)
            if M["m00"] != 0:
                redCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Centroid

                if radius > 10:
                    cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv.circle(frame, redCenter, 5, (0, 0, 255), -1)
                    
def blue():
        mask = cv.inRange(hsv, blueLower, blueUpper)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)

        # Find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        blueCenter = None

        # Proceed when a contour is found
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)  # Find the largest contour in the mask
            ((x, y), radius) = cv.minEnclosingCircle(c)  # Compute the minimum enclosing circle
            M = cv.moments(c)
            if M["m00"] != 0:
                blueCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Centroid

                if radius > 10:
                    cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv.circle(frame, blueCenter, 5, (0, 0, 255), -1)
  
# Construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Lower and upper boundaries of the "orange" color in HSV color space
orangeLower = (10, 100, 100)
orangeUpper = (30, 255, 255)

purpleLower = (140, 50, 50)
purpleUpper = (160, 255, 255)

redLower = (160, 100, 100)
redUpper = (179, 255, 255)

blueLower = (90, 50, 50)
blueUpper = (130, 255, 255)

pts = deque(maxlen=args["buffer"])

# Video path not supplied, grab reference to Webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv.VideoCapture(args["video"])

while True:
    frame = vs.read()  # Grab current frame
    frame = frame[1] if args.get("video", False) else frame  # Handle frame from VideoCapture or VideoStream

    # Video ended or no frame received
    if frame is None:
        break

    # Resize frame, blur it, and convert it to HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
              
    orange()
    purple()
    red()
    blue()
                
    # if orange() is not None:
    #     frame_width = frame.shape[1]
    #     middle_x = frame_width // 2
    #     if orange[0] > middle_x - RIGHT_CENTER_X_THRESHOLD and orange[0] < middle_x + LEFT_CENTER_X_THRESHOLD:
    #             sound_thread = Thread(target=play_sound)
    #             sound_thread.start()
                
    # if purple() is not None:
    #     frame_width = frame.shape[1]
    #     middle_x = frame_width // 2
    #     if purple[0] > middle_x - RIGHT_CENTER_X_THRESHOLD and purple[0] < middle_x + LEFT_CENTER_X_THRESHOLD:
    #             sound_thread = Thread(target=play_sound)
    #             sound_thread.start()
            
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv.destroyAllWindows()    

    