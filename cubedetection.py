from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
from matplotlib import pyplot as plt

from pythonosc.udp_client import SimpleUDPClient

ip = "10.100.1.128"
port = 54345

def yellow():
    generalcubefinder(frame, yellowLower, yellowUpper, 'yellow')
    
def red():
    generalcubefinder(frame, redLower, redUpper, 'red')

def blue():
    generalcubefinder(frame, blueLower, blueUpper, 'blue')

def green():
    generalcubefinder(frame, greenLower, greenUpper, 'green')

def generalcubefinder(frame, lwr_iro_bnd, upr_iro_bnd, color_name):
    # Construct a mask for purple color, perform dilations and erosions to remove blobs
    mask = cv.inRange(hsv, lwr_iro_bnd, upr_iro_bnd)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    ret, thresh = cv.threshold(mask, 50, 255, 0)

    # Find contours in the mask and initialize the current (x, y) center of the ball
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("Number of Countours:", len(contours))
    
    for cnt in contours:
        x1, y1 = cnt[0][0]
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        if len(approx) >= 4:
            ind = 0
            x, y, w, h = cv.boundingRect(cnt)
            frame = cv.drawContours(frame, [cnt], -1, (0, 255, 255), 3)
            cv.putText(frame, color_name + "Square", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # client.send_message("square", [color_name, ind, x, y])  # Send message with int, float and string 
            # ind = ind + 1

# Construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Lower and upper boundaries of the "yellow" color in HSV color space
yellowLower = (25, 100, 100)
yellowUpper = (35, 255, 255)

# Lower and upper boundaries of the "red" color in HSV color space
redLower = (0, 100, 100)
redUpper = (10, 255, 255)

# Lower and upper boundaries of the "light blue" color in HSV color space
blueLower = (90, 100, 100)
blueUpper = (110, 255, 255)

# Lower and upper boundaries of the "green" color in HSV color space
greenLower = (60, 100, 100)
greenUpper = (90, 255, 255)

pts = deque(maxlen=args["buffer"])

# Video path not supplied, grab reference to Webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv.VideoCapture(args["video"])

client = SimpleUDPClient(ip, port)  # Create client

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
              
    ret = None  
    hierarchy = None  
    x = None  
    y = None  
    ratio = None  
    
    yellow()
    red()
    blue()
    green()
            
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv.destroyAllWindows()
