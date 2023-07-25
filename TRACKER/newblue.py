from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
import time
from threading import Thread
from playsound import playsound

# Construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Lower and upper boundaries of the "purple" color in HSV color space
blueLower = (90, 50, 50)
blueUpper = (130, 255, 255)
pts = deque(maxlen=args["buffer"])

# Video path not supplied, grab reference to Webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv.VideoCapture(args["video"])

xprev = 0
yprev = 0
radiusprev = 0
k = 20
lenprev = 0

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

    # Construct a mask for purple color, perform dilations and erosions to remove blobs
    mask = cv.inRange(hsv, blueLower, blueUpper)
    mask = cv.erode(mask, None, iterations=2)
    # mask = cv.Canny(mask, 500, 400)
    mask = cv.dilate(mask, None, iterations=2)

    # Find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("Number of Countours:", len(cnts))
    center = None
    
    for cnt in cnts:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        # Proceed when a contour is found
        # c = max(cnts, key=cv.contourArea)  # Find the largest contour in the mask
        ((x, y), radius) = cv.minEnclosingCircle(cnt)  # Compute the minimum enclosing circle
        M = cv.moments(cnt)
        if M["m00"] != 0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        if radius > 30:
        
            if lenprev == 0:
                lennow = len(approx)
            else:
                lennow = int((len(approx) + k * lenprev)/(1 + k))
        
                # putting shape name at center of each shape
            if lennow == 3:
                cv.putText(frame, 'Triangle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv.drawContours(frame, [cnt], -1, (0,255,255), 3)
        
            elif lennow >= 4 and len(approx) <= 10:
                    
                if radiusprev != 0:
                    xnow = int((x + k * xprev)/(1 + k))
                    ynow = int((y + k * yprev)/(1 + k))
                        
                    print(xnow, ynow)
                        
                else: 
                    xnow = x
                    ynow = y
                        
                cv.putText(frame, 'Quadrilateral', (xnow, ynow), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv.drawContours(frame, [cnt], -1, (0,255,255), 3)
                        
                xprev = xnow
                yprev = ynow
                radiusprev = radius
                lenprev = lennow
                    
            else:
                cv.putText(frame, 'circle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Centroid

                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)
                
        else:
            radiusprev = 0

        # pts.appendleft(center)

        # for i in range(1, len(pts)):
        #     if pts[i - 1] is None or pts[i] is None:
        #         continue

        #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #     cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        
    cv.imshow("Frame", frame)
    cv.imshow("Mask", mask)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv.destroyAllWindows()