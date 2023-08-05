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


def create_windows_and_trackbars(redLower, redUpper, blueLower, blueUpper, greenLower, greenUpper, yellowLower, yellowUpper, hueShift, satL, satH, lumL, lumH, smallSizeL, smallSizeH, bigSizeL, bigSizeH):
    
    cv.namedWindow('Frame')
    cv.moveWindow('Frame', 200, 0)

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
    cv.createTrackbar('lumL', 'Trackbars', int(lumL), 255, nothing)
    cv.createTrackbar('lumH', 'Trackbars', int(lumH), 255, nothing)    
    cv.createTrackbar('smallSizeL', 'Trackbars', int(smallSizeL), 255, nothing)
    cv.createTrackbar('smallSizeH', 'Trackbars', int(smallSizeH), 255, nothing)
    cv.createTrackbar('bigSizeL', 'Trackbars', int(bigSizeL), 255, nothing)
    cv.createTrackbar('bigSizeH', 'Trackbars', int(bigSizeH), 255, nothing)        
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
    print('saving',boundaries)
    with open(filename, 'wb') as f:
        pickle.dump(boundaries, f)
