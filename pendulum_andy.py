import cv2 as cv
import numpy as np
from threading import Thread
from pygame import mixer
import pygame
import rtmidi

midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
print(available_ports)
# Attempt to connect to the IAC bus
if available_ports:
    for i, port in enumerate(available_ports):
        if 'IAC Driver Bus 1' in port:  # replace 'IAC Driver Bus 1' with your IAC bus name
            midiout.open_port(i)
            print('Connected to port:', port)
            break
    else:
        print('IAC bus not found. Opening a virtual port instead.')
        midiout.open_virtual_port('My virtual output')
else:
    print('No available MIDI ports. Opening a virtual port.')
    midiout.open_virtual_port('My virtual output')




# Constants
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
MIN_BALL_RADIUS = 6
MAX_BALL_RADIUS = 42
RIGHT_CENTER_X_THRESHOLD = (VIDEO_WIDTH / 2) + 50  # Adjust this value based on your video frame width
LEFT_CENTER_X_ThRESHOLD = (VIDEO_WIDTH / 2) - 50
DIST_THRESHOLD = 40  # Threshold for distance comparison
FILTER_LENGTH = 3  # Length of moving average filter


class MutableInt:
    def __init__(self, value=0):
        self.value = value

hough_param1 = MutableInt(52)
hough_param2 = MutableInt(27)


# Initialize Pygame
pygame.init()

# Create a Pygame window
win = pygame.display.set_mode((90, 222))
# Define button properties
button_width = 80
button_height = 40

# Define buttons
buttons = [
    pygame.Rect(5, 5, button_width, button_height),    # Increase Param1 Button
    pygame.Rect(5, 55, button_width, button_height),   # Decrease Param1 Button
    pygame.Rect(5, 105, button_width, button_height),    # Increase Param2 Button
    pygame.Rect(5, 155, button_width, button_height),   # Decrease Param2 Button
]

actions = [
    lambda: setattr(hough_param1, "value", hough_param1.value + 1),  # Increase hough_param1
    lambda: setattr(hough_param1, "value", hough_param1.value - 1),  # Decrease hough_param1
    lambda: setattr(hough_param2, "value", hough_param2.value + 1),  # Increase hough_param2
    lambda: setattr(hough_param2, "value", hough_param2.value - 1),  # Decrease hough_param2
]

for i, button in enumerate(buttons):
    pygame.draw.rect(win, (33, 111, 111), button)  # draw button

pygame.display.flip()


videoCapture = cv.VideoCapture(0)
# Reduce the resolution
videoCapture.set(cv.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
videoCapture.set(cv.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

prev_x = None
prev_avg_dx = None
avg_dx = 1
dx_values = []
dx_values_since_change = []
prev_sign = None




mixer.init(channels=64)
sound = mixer.Sound('Piano_C3.wav')
channel_number = 0

def play_sound():
    global channel_number
    mixer.Channel(channel_number).play(sound)
    channel_number = (channel_number + 1) % mixer.get_num_channels()
    note_on = [0x90, 60, 112]  # channel 1, middle C, velocity 112
    note_off = [0x80, 60, 0]
    midiout.send_message(note_off)
    midiout.send_message(note_on)

    
while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (11, 11), 0)

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, minDist=100, param1=hough_param1.value, param2=hough_param2.value, minRadius=MIN_BALL_RADIUS, maxRadius=MAX_BALL_RADIUS)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            else:
                dist = np.linalg.norm(chosen[:2] - i[:2])
                if dist <= DIST_THRESHOLD:
                    chosen = i

        center_x = chosen[0]
        radius = chosen[2]

        if prev_x is not None:
            dx = int(center_x) - int(prev_x)
            dx_values.append(dx)
            dx_values_since_change.append(dx)
            if len(dx_values) > FILTER_LENGTH:  # If the length exceeds the filter length, remove the oldest value
                dx_values.pop(0)
            avg_dx = np.mean(dx_values)  # Calculate average dx

            # Check if the circle changes direction
            current_sign = np.sign(avg_dx)
            if prev_sign is not None and current_sign != prev_sign:
                peak_speed = max([abs(dx) for dx in dx_values_since_change])
                dx_values_since_change = []  # Reset the dx_values_since_change
                play_sound()

            prev_sign = current_sign

        prev_x = center_x


        # Draw circle and center point on the frame
        cv.circle(frame, (chosen[0], chosen[1]), radius, (0, 100, 100), 3)
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)

        cv.putText(frame, 'dx: {:.2f}'.format(avg_dx), (10, VIDEO_HEIGHT - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)


    cv.imshow("circles", frame)

    key = cv.waitKey(1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos  # gets mouse position
            # check if any buttons have been clicked
            for button in buttons:
                if button.collidepoint(mouse_pos):
                    index = buttons.index(button)
                    actions[index]()
                    print(hough_param1.value, hough_param2.value)

videoCapture.release()
cv.destroyAllWindows()