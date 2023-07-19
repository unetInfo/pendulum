import cv2
import numpy as np

# Capture a frame from the camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Convert the frame to HSV
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

print('NOTE:  to quit, click on the video window and type any key...')

# Function to handle mouse click event
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Define the size of the region to sample around the clicked point
        region_size = 5
        # Ensure the region stays within the frame boundaries
        x_start = max(x - region_size//2, 0)
        x_end = min(x + region_size//2, frame.shape[1] - 1)
        y_start = max(y - region_size//2, 0)
        y_end = min(y + region_size//2, frame.shape[0] - 1)
        
        region = hsv_frame[y_start:y_end, x_start:x_end]
        average_color = np.mean(region, axis=(0, 1))
        
        print('HSV value at this region is: ', average_color, x, y)

# Set up window for user interaction
cv2.namedWindow('image')
cv2.setMouseCallback('image', pick_color)

# Display the image and wait for a key press
cv2.imshow('image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
