
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
from pythonosc.udp_client import SimpleUDPClient
from tracker import EuclideanDistTracker

# Construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
# ap.add_argument("-i", "--ip", required=True, help="IP address of the machine running Golden")
args = vars(ap.parse_args())

# TamperLab Mac Laptop
# python doublependulum.py --ip 10.100.1.128

# Andy's Mac Studio
# python3 doublependulum.py --ip 192.168.1.107

port = 54345
ip_addr = "10.100.1.126"

def yellow():
    generalSpherefinder(yellowLower, yellowUpper ,'yellow', detections)

def green():
    generalSpherefinder(greenLower, greenUpper, 'green', detections)

def red():
    generalSpherefinder(redLower, redUpper, 'red', detections)

def blue():
    generalSpherefinder(blueLower, blueUpper ,'blue', detections)

def generalSpherefinder(lwr_iro_bnd, upr_iro_bnd, color_name, detections):

    # Construct a mask for purple color, perform dilations and erosions to remove blobs
    mask = cv.inRange(hsv, lwr_iro_bnd, upr_iro_bnd)
    mask = cv.erode(mask, None, iterations=2)
    # mask = cv.Canny(mask, 500, 400)
    mask = cv.dilate(mask, None, iterations=2)

    # Find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("Number of Countours:", len(cnts))
    center = None
    # detections = []

    for cnt in cnts:
        foundShape = ""
        approx = cv.approxPolyDP(cnt, 0.04 * cv.arcLength(cnt, True), True)
        # Proceed when a contour is found
        # c = max(cnts, key=cv.contourArea)  # Find the largest contour in the mask
        ((x, y), radius) = cv.minEnclosingCircle(cnt)  # Compute the minimum enclosing circle
        M = cv.moments(cnt)
        if M["m00"] != 0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        if radius > 20:

            w = radius
            h = radius
            ind = 0

            # if lenprev == 0:
            #     lennow = len(approx)
            # else:
            #     lennow = int((len(approx) + k * lenprev)/(1 + k))

                # putting shape name at center of each shape
            if len(approx) == 3:
                foundShape = 'Triangle'
                cv.putText(frame, color_name + foundShape, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv.drawContours(frame, [cnt], -1, (0,255,255), 3)
                # client.send_message("SOMeCOR", ["TRIANGLE", color_name, ind, center[0], center[1]])  # Send message with int, float and string
                # ind = ind + 1

            elif len(approx) == 4 and len(approx) <= 10:
                if radius <= 30:
                # if radiusprev != 0:
                #     xnow = int((x + k * xprev)/(1 + k))
                #     ynow = int((y + k * yprev)/(1 + k))

                #     print(xnow, ynow)

                # else:
                #     xnow = x
                #     ynow = y
                    foundShape = 'SmallQuad'
                    cv.putText(frame, color_name + foundShape, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    #osc_hose.send_message("SOMeCOR", [foundShape, color_name, ind, center[0], center[1]])  # Send message with int, float and string
                    # ind = ind + 1

                else:
                    foundShape = 'BigQuad'
                    cv.putText(frame, color_name + foundShape, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    #osc_hose.send_message("SOMeCOR", [foundShape, color_name, ind, center[0], center[1]])  # Send message with int, float and string
                    # ind = ind + 1

                cv.drawContours(frame, [approx], -1, (0,255,255), 3)


                # xprev = xnow
                # yprev = ynow
                # radiusprev = radius
                # lenprev = lennow

            else:
                foundShape = 'Circle'
                cv.putText(frame, color_name + foundShape, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Centroid

                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)
                #osc_hose.send_message("SOMeCOR", [foundShape, color_name, ind, center[0], center[1]])  # Send message with int, float and string
                # ind = ind + 1

            detections.append([x, y, w, h, foundShape, color_name])



        # else:
        #     radiusprev = 0

                    #

    return detections

    # detections.remove([x, y, w, h])


# Lower and upper boundaries of the "orange" color in HSV color space
yellowLower = (20, 50, 50)
yellowUpper = (35, 255, 255)

greenLower = (40, 50, 50)
greenUpper = (80, 255, 255)

redLower = (0, 100, 100)
redUpper = (10, 255, 255)

blueLower = (90, 50, 50)
blueUpper = (130, 255, 255)

pts = deque(maxlen=args["buffer"])

# Video path not supplied, grab reference to Webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv.VideoCapture(args["video"])

#osc_hose = SimpleUDPClient(args["ip"], port)  # Create client
osc_hose = SimpleUDPClient(ip_addr, port)  # Create client

tracker = EuclideanDistTracker()
detections = []

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

    yellow()
    green()
    red()
    blue()

    objectIDs = tracker.update(detections)
    for objectID in objectIDs:
        x, y, w, h, shap, farbe, id = objectID
        cv.putText(frame, str(id),(x,y-15),  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        osc_hose.send_message("/SOMeCOR", [id, shap, farbe, x, y])  # Send message with int, float and string
        detections.remove([x, y, w, h, shap, farbe])

    cv.imshow("Mask", hsv)
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv.destroyAllWindows()
