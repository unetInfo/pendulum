
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
from pythonosc.udp_client import SimpleUDPClient
from tracker import EuclideanDistTracker
import time
from somecor_interface2 import find_cameras, create_windows_and_trackbars, read_local_boundaries, save_local_boundaries

# - Red: 0 or 360 degrees. For a range, you might use 330 to 30 degrees as the lower and upper bounds, respectively.
# - Green: 120 degrees. For a range, you might use 90 to 150 degrees.
# - Blue: 240 degrees. For a range, you might use 210 to 270 degrees.
# - Yellow: 60 degrees. For a range, you might use 30 to 90 degrees.

# Remember to halve these values when translating to the Hue values for OpenCV's 0-179 range. For example, red would range from about 165 to 15 in OpenCV's scale.

# Define the default boundaries in 180 space
default_boundaries = {
    'yellowLower': (15, 50, 50),
    'yellowUpper': (45, 255, 255),
    'greenLower': (47, 50, 50),
    'greenUpper': (107, 255, 255),
    'redLower': (0, 100, 100),
    'redUpper': (14, 255, 255),
    'blueLower': (108, 50, 50),
    'blueUpper': (135, 255, 255),
    'hueShift': 0,
    'satL': 50,
    'satH': 255,
    'lumL': 50,
    'lumH': 255,
    'smallSizeL': 20,
    'smallSizeH': 30,
    'bigSizeL': 20,
    'bigSizeH': 30,
    'approxPoly': 0.04, 
    'lenApproxH': 6,}

# Path to the saved boundaries
filename = 'boundaries.pkl'
boundaries = read_local_boundaries(filename) or default_boundaries

yellowLower = boundaries['yellowLower']
yellowUpper = boundaries['yellowUpper']
greenLower = boundaries['greenLower']
greenUpper = boundaries['greenUpper']
redLower = boundaries['redLower']
redUpper = boundaries['redUpper']
blueLower = boundaries['blueLower']
blueUpper = boundaries['blueUpper']
hueShift = boundaries['hueShift']
satL = boundaries['satL']
satH = boundaries['satH']
lumL = boundaries['lumL']
lumH = boundaries['lumH']
smallSizeL = boundaries['smallSizeL']
smallSizeH = boundaries['smallSizeH']
bigSizeL = boundaries['bigSizeL']
bigSizeH = boundaries['bigSizeH']
approxPoly = ['approxPoly']
lenApproxH = ['lenApproxH']

# Construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
# ap.add_argument("-i", "--ip", required=True, help="IP address of the machine running Golden")
args = vars(ap.parse_args())

port = 54345
#ip_addr = "10.100.1.126"
#ip_addr = "192.168.1.107"
ip_addr = "127.0.0.1"

available_cameras = find_cameras(2)
print('Available cameras:', available_cameras)

create_windows_and_trackbars(redLower, redUpper, blueLower, blueUpper, greenLower, greenUpper, yellowLower, yellowUpper, hueShift, satL, satH, lumL, lumH, smallSizeL, smallSizeH, bigSizeL, bigSizeH, approxPoly, lenApproxH)

def yellow():
    generalObjectfinder(yellowLower, yellowUpper ,'yellow', detections)

def green():
    generalObjectfinder(greenLower, greenUpper, 'green', detections)

def red():
    generalObjectfinder(redLower, redUpper, 'red', detections)

def blue():
    generalObjectfinder(blueLower, blueUpper ,'blue', detections)

def nothing(x):
    pass

def order_points(pts):
    # Sort the points based on their x-coordinate
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # Grab the left-most (smallest x) and right-most (largest x) points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # Now, sort the left-most coordinates according to their y-coordinates so we can grab the top-left and bottom-left points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # Now that we have the top-left coordinate, use it as an anchor to calculate the Euclidean distance between the top-left and right-most points; by the Pythagorean theorem, the point with the largest distance will be our bottom-right point
    D = np.linalg.norm(right_most - tl, axis=1)
    (br, tr) = right_most[np.argsort(D)[::-1], :]

    # Return the coordinates in top-left, top-right, bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def generalObjectfinder(lwr_iro_bnd, upr_iro_bnd, color_name, detections):

    # Construct a mask for purple color, perform dilations and erosions to remove blobs
    mask = cv.inRange(hsv, lwr_iro_bnd, upr_iro_bnd)
    mask = cv.erode(mask, None, iterations=2)
    # mask = cv.Canny(mask, 500, 400)
    mask = cv.dilate(mask, None, iterations=2)
    
    cv.imshow("Mask" + color_name, mask)

    # Find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #print("Number of Countours:", len(cnts))
    center = None
    # detections = []

    for cnt in cnts:
        foundShape = ""
        rotation_angle = 0
        # area = cv.contourArea(cnt)
        approx = cv.approxPolyDP(cnt, (approxPoly/100) * cv.arcLength(cnt, True), True)
        # Proceed when a contour is found
        # c = max(cnts, key=cv.contourArea)  # Find the largest contour in the mask
        ((x, y), radius) = cv.minEnclosingCircle(cnt)  # Compute the minimum enclosing circle
        M = cv.moments(cnt)
        if M["m00"] != 0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        if radius > smallSizeL and radius < bigSizeH:
            w = radius
            h = radius
            ind = 0

            if len(approx) >= 4 and len(approx) <= lenApproxH:

                # Order the points in the contour and unpack them
                rect = cv.minAreaRect(approx)
                box = cv.boxPoints(rect)
                box = np.array(box, dtype="int")

                # Order the points in the rectangle [top-left, top-right, bottom-right, bottom-left]
                box = order_points(box)

                # Compute the deltas to get the vectors representing the rectangle edges
                (tl, tr, br, bl) = box
                dX = tr[0] - tl[0]
                dY = tr[1] - tl[1]

                # Compute the angle, convert it to degrees and normalize it
                angle = np.degrees(np.arctan2(dY, dX))
                angle = angle if angle > 0 else angle + 360

                rotation_angle = angle

                if radius <= smallSizeH:
                
                    foundShape = 'SmallQuad'
                    cv.putText(frame, color_name + foundShape, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # ind = ind + 1

                elif radius >= bigSizeL:
                    foundShape = 'BigQuad'
                    cv.putText(frame, color_name + foundShape, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # ind = ind + 1

                cv.drawContours(frame, [approx], -1, (0,255,255), 3)

                detections.append([x, y, w, h, foundShape, color_name])

            else:
                foundShape = 'Circle'
                cv.putText(frame, color_name + foundShape, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  # Centroid

                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)
                # # ind = ind + 1
                
                detections.append([x, y, w, h, foundShape, color_name])

    return detections

def on_mouse_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        region_size = 5
        x_start = max(x - region_size//2, 0)
        x_end = min(x + region_size//2, frame.shape[1] - 1)
        y_start = max(y - region_size//2, 0)
        y_end = min(y + region_size//2, frame.shape[0] - 1)
        
        region = hsv[y_start:y_end, x_start:x_end]
        average_color = np.mean(region, axis=(0, 1))
        
        print('HSV value at this region is: ', average_color[0] * 2, average_color[1], average_color[2])

cv.setMouseCallback('Frame', on_mouse_event)

pts = deque(maxlen=args["buffer"])

# Video path not supplied, grab reference to Webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv.VideoCapture(args["video"])

# Allow the camera sensor to warm up
time.sleep(1.5)

#osc_hose = SimpleUDPClient(args["ip"], port)  # Create client
osc_hose = SimpleUDPClient(ip_addr, port)  # Create client

tracker = EuclideanDistTracker()
detections = []

# Create trackbars for color change
# For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
# If you're working in a different color space, the ranges might be different.
cv.createTrackbar('redL', 'Trackbars', int(redLower[0] * 2), 359, nothing)
cv.createTrackbar('redH', 'Trackbars', int(redUpper[0] * 2), 359, nothing)
cv.createTrackbar('blueL', 'Trackbars', int(blueLower[0] * 2), 359, nothing)
cv.createTrackbar('blueH', 'Trackbars', int(blueUpper[0] * 2), 359, nothing)
cv.createTrackbar('greenL', 'Trackbars', int(greenLower[0] * 2), 359, nothing)
cv.createTrackbar('greenH', 'Trackbars', int(greenUpper[0] * 2), 359, nothing)
cv.createTrackbar('yellowL', 'Trackbars', int(yellowLower[0] * 2), 359, nothing)
cv.createTrackbar('yellowH', 'Trackbars', int(yellowUpper[0] * 2), 359, nothing)
cv.createTrackbar('hueShift', 'Trackbars', int(hueShift + 60), 120, nothing)
cv.createTrackbar('Kill','Trackbars', 0, 1,nothing)

frame_counter = 0
update_interval = 10  # Adjust this to change how often the trackbars are read

# loop over the frames from the video stream
while True:
    frame = vs.read()  # Grab current frame
    frame = frame[1] if args.get("video", False) else frame  # Handle frame from VideoCapture or VideoStream

    # Video ended or no frame received
    if frame is None:
        break

    if frame_counter % update_interval == 0:
        # Get current values of the trackbar positions
        hueShift = (cv.getTrackbarPos('hueShift', 'Trackbars') - 60) / 2
        satL = cv.getTrackbarPos('satL', 'Trackbars')
        satH = cv.getTrackbarPos('satH', 'Trackbars')
        lumL = cv.getTrackbarPos('lumL', 'Trackbars')
        smallSizeL = cv.getTrackbarPos('smallSizeL', 'Trackbars')
        smallSizeH = cv.getTrackbarPos('smallSizeH', 'Trackbars')
        bigSizeL = cv.getTrackbarPos('bigSizeL', 'Trackbars')
        bigSizeH = cv.getTrackbarPos('bigSizeH', 'Trackbars')
        approxPoly = cv.getTrackbarPos('approxPoly', 'Trackbars')
        lenApproxH = cv.getTrackbarPos('lenApproxH', 'Trackbars')

        lumH = cv.getTrackbarPos('lumH', 'Trackbars')
        redL = hueShift + cv.getTrackbarPos('redL', 'Trackbars') / 2
        redH = hueShift + cv.getTrackbarPos('redH', 'Trackbars') / 2
        blueL = hueShift + cv.getTrackbarPos('blueL', 'Trackbars') / 2
        blueH = hueShift + cv.getTrackbarPos('blueH', 'Trackbars') / 2
        greenL = hueShift + cv.getTrackbarPos('greenL', 'Trackbars') / 2
        greenH = hueShift + cv.getTrackbarPos('greenH', 'Trackbars') / 2
        yellowL = hueShift + cv.getTrackbarPos('yellowL', 'Trackbars') / 2
        yellowH = hueShift + cv.getTrackbarPos('yellowH', 'Trackbars') / 2
        yellowLower = (yellowL, satL, lumL)
        yellowUpper = (yellowH, satH, lumH)
        greenLower = (greenL, satL, lumL)
        greenUpper = (greenH, satH, lumH)
        redLower = (redL, satL, lumL)
        redUpper = (redH, satH, lumH)
        blueLower = (blueL, satL, lumL)
        blueUpper = (blueH, satH, lumH)
        if cv.getTrackbarPos('Kill', 'Trackbars') > 0:
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
    oscs = []
    for objectID in objectIDs:
        x, y, w, h, shap, farbe, id = objectID
        oscs.append([id, shap, farbe, x, y])
        cv.putText(frame, str(id),(x,y-15),  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        detections.remove([x, y, w, h, shap, farbe])

    osc_hose.send_message("/SOMeCOR", oscs)  # Send message with int, float and string

    cv.imshow("Mask", hsv)
    cv.imshow("Frame", frame)

    # even though the keyboard reading doesnt work in osx, it is needed to keep the window happy somehow
    key = cv.waitKey(21) & 0xFF
    if key == ord("q"):
       break
   
print('Graceful end')

# bundle up the settings so we can save them
boundaries = {
    'yellowLower': yellowLower,
    'yellowUpper': yellowUpper,
    'greenLower': greenLower,
    'greenUpper': greenUpper,
    'redLower': redLower,
    'redUpper': redUpper,
    'blueLower': blueLower,
    'blueUpper': blueUpper,
}

# Perform the hueShift operation on the boundaries
for key, value in boundaries.items():
    boundaries[key] = (value[0] - hueShift, value[1], value[2])

# Add the non H values to the boundaries dictionary
boundaries['hueShift'] = hueShift
boundaries['satL'] = satL
boundaries['satH'] = satH
boundaries['lumL'] = lumL
boundaries['lumH'] = lumH
boundaries['smallSizeL'] = smallSizeL
boundaries['smallSizeH'] = smallSizeH
boundaries['bigSizeL'] = bigSizeL
boundaries['bigSizeH'] = bigSizeH

save_local_boundaries(boundaries, filename)

if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv.destroyAllWindows()
