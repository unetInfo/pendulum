from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
import time
from threading import Thread
from playsound import playsound

from pythonosc.udp_client import SimpleUDPClient

# Construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
ap.add_argument("-i", "--ip", required=True, help="IP address of the machine running Golden")
args = vars(ap.parse_args())


# TamperLab Mac Laptop
# python doublependulum.py --ip 10.100.1.128

# Andy's Mac Studio 
# python3 doublependulum.py --ip 192.168.1.107


port = 54345

# Constants
RIGHT_CENTER_X_THRESHOLD = 280  # Adjust this value based on your video frame width
LEFT_CENTER_X_THRESHOLD = 320
DIST_THRESHOLD = 100  # Threshold for distance comparison

def play_sound():
    playsound('Piano_C3.mp3')

def orange():
    generalSpherefinder(orangeLower, orangeUpper ,'orange')
    
def purple():
    generalSpherefinder(purpleLower, purpleUpper, 'purple')
    
def red():
    generalSpherefinder(redLower, redUpper, 'red')
    
def blue():
    generalSpherefinder(blueLower, blueUpper ,'blue')

def green():
    generalSpherefinder(greenLower, greenUpper ,'green')

def yellow():
    generalSpherefinder(yellowLower, yellowUpper ,'yellow')

def generalSpherefinder(lwr_iro_bnd, upr_iro_bnd, color_name):
        mask = cv.inRange(hsv, lwr_iro_bnd, upr_iro_bnd)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)

        # Find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # Proceed when a contour is found
        if len(cnts) > 0:
            ind = 0
            c = max(cnts, key=cv.contourArea)  # Find the largest contour in the mask
            ((x, y), radius) = cv.minEnclosingCircle(c)  # Compute the minimum enclosing circle
            M = cv.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Centroid

                if radius > 10:
                    cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv.circle(frame, center, 5, (0, 0, 255), -1)
                    client.send_message("/pendulum", [color_name, ind, center[0], center[1]])  # Send message with int, float and string 
                    ind = ind + 1
  


# Lower and upper boundaries of the "orange" color in HSV color space
orangeLower = (10, 100, 100)
orangeUpper = (30, 255, 255)

purpleLower = (140, 50, 50)
purpleUpper = (160, 255, 255)

redLower = (160, 100, 100)
redUpper = (179, 255, 255)

blueLower = (90, 50, 20)
blueUpper = (130, 255, 255)

greenLower = (36, 50, 50)
greenUpper = (86, 255, 255)

yellowLower = (15, 50, 50)
yellowUpper = (35, 255, 255)


pts = deque(maxlen=args["buffer"])

# Video path not supplied, grab reference to Webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv.VideoCapture(args["video"])

client = SimpleUDPClient(args["ip"], port)  # Create client

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
              
    # orange()
    # purple()
    red()
    blue()
    green()
    yellow()
             
    # client.send_message("/pendulum", [1, 2., "hello"])  # Send message with int, float and string   
    
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

    
