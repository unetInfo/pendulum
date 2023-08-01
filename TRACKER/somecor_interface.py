import cv2 as cv
import numpy as np
import pickle
import os

def nothing(x):
    pass

def find_cameras(max_cameras_to_test=2):
    available_cameras = []
    for i in range(max_cameras_to_test):
        cap = cv.VideoCapture(i)
        if cap is None or not cap.isOpened():
            pass
        else:
            print('Camera available:', i)
            available_cameras.append(i)
        cap.release()
    return available_cameras

def on_mouse_event(event, x, y, flags, param):
    global checkbox_status, checkbox_img
    if event == cv.EVENT_LBUTTONUP:
        region_size = 5
        x_start = max(x - region_size//2, 0)
        x_end = min(x + region_size//2, frame.shape[1] - 1)
        y_start = max(y - region_size//2, 0)
        y_end = min(y + region_size//2, frame.shape[0] - 1)
        
        region = hsv[y_start:y_end, x_start:x_end]
        average_color = np.mean(region, axis=(0, 1))
        
        print('HSV value at this region is: ', average_color)

def create_windows_and_trackbars(redLower, redUpper, blueLower, blueUpper, greenLower, greenUpper, yellowLower, yellowUpper, hueShift, satL, satH, lumL, lumH):
    cv.namedWindow('Frame')
    cv.moveWindow('Frame', 200, 0)
    cv.setMouseCallback('Frame', on_mouse_event)

    cv.namedWindow('Mask')
    cv.moveWindow('Mask', 200, 370)

    cv.namedWindow('Trackbars')

    cv.createTrackbar('redL', 'Trackbars', int(redLower[0] * 2), 359, nothing)
    cv.createTrackbar('redH', 'Trackbars', int(redUpper[0] * 2), 359, nothing)
    cv.createTrackbar('yellowL', 'Trackbars', int(yellowLower[0] * 2), 359, nothing)
    cv.createTrackbar('yellowH', 'Trackbars', int(yellowUpper[0] * 2), 359, nothing)
    cv.createTrackbar('greenL', 'Trackbars', int(greenLower[0] * 2), 359, nothing)
    cv.createTrackbar('greenH', 'Trackbars', int(greenUpper[0] * 2), 359, nothing)
    cv.createTrackbar('blueL', 'Trackbars', int(blueLower[0] * 2), 359, nothing)
    cv.createTrackbar('blueH', 'Trackbars', int(blueUpper[0] * 2), 359, nothing)
    cv.createTrackbar('hueShift', 'Trackbars', int(hueShift + 60), 120, nothing)
    cv.createTrackbar('satL', 'Trackbars', int(satL), 255, nothing)
    cv.createTrackbar('satH', 'Trackbars', int(satH), 255, nothing)
    cv.createTrackbar('lumL', 'Trackbars', int(satL), 255, nothing)
    cv.createTrackbar('lumH', 'Trackbars', int(satH), 255, nothing)    
    cv.createTrackbar('Kill','Trackbars',0,1,nothing)

def read_local_boundaries(filename = 'boundaries.pkl'):
    # If the file exists, load the boundaries from disk
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            boundaries = pickle.load(f)
            print('loaded',boundaries)
    else:
        boundaries = None
    return boundaries

def save_local_boundaries(boundaries, filename = 'boundaries.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(boundaries, f)
