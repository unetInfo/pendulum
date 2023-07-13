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
MIN_BALL_RADIUS = 11
MAX_BALL_RADIUS = 21
MIN_DIST_BETWEEN_CIRCLES = 140
RIGHT_CENTER_X_THRESHOLD = (VIDEO_WIDTH / 2) + 50  # Adjust this value based on your video frame width
LEFT_CENTER_X_ThRESHOLD = (VIDEO_WIDTH / 2) - 50
DIST_THRESHOLD = 40  # Threshold for distance comparison
FILTER_LENGTH = 6  # Length of moving average filter
MAX_SIMULTANEOUS_CIRCLES = 1

mixer.init(channels=64)
sounds = [mixer.Sound('Piano_C3.wav'), mixer.Sound('Piano_G3.wav')]
channel_number = 0

def play_sound(index, direction, speed):
    global channel_number
    velocity = np.clip(1 + (speed * 0.4), 1, 127)
    volume_level = velocity / 127
    index_offset = index + (0 if direction == -1 else 1)
    #print(index_offset, direction, speed, velocity, volume_level)
    sounds[index_offset].set_volume(volume_level)
    mixer.Channel(channel_number).play(sounds[index_offset])
    channel_number = (channel_number + 1) % mixer.get_num_channels()
    step = 60 + (index_offset * 7)
    note_on = [0x90, step, 64 + (velocity / 2)]
    note_off = [0x80, step, 0]
    midiout.send_message(note_off)
    midiout.send_message(note_on)


class MutableInt:
    def __init__(self, value=0):
        self.value = value

hough_param1 = MutableInt(106)
hough_param2 = MutableInt(26)


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


# Initialize previous and current position info for two pendulums
prev_x = [None, None]
prev_avg_dx = [None, None]
avg_dx = [1, 1]
dx_values = [[], []]
dx_values_since_change = [[], []]
prev_sign = [None, None]

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)
    #blurFrame = grayFrame

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, minDist=MIN_DIST_BETWEEN_CIRCLES, param1=hough_param1.value, param2=hough_param2.value, minRadius=MIN_BALL_RADIUS, maxRadius=MAX_BALL_RADIUS)

    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Get the two closest circles to the last position of each pendulum
        if len(circles[0]) >= MAX_SIMULTANEOUS_CIRCLES:
            if prev_x[0] is None and prev_x[1] is None:  # If it's the first frame
                pendulums = circles[0, :MAX_SIMULTANEOUS_CIRCLES]  # Just take the best MAX_SIMULTANEOUS_CIRCLES
            else:
                pendulums = sorted(circles[0], key=lambda c: min(
                    (c[0] - prev_x[0]) ** 2 if prev_x[0] is not None else float('inf'),
                    (c[0] - prev_x[1]) ** 2 if prev_x[1] is not None else float('inf'))
                )[:MAX_SIMULTANEOUS_CIRCLES]        
            for i, chosen in enumerate(pendulums):
                center_x = chosen[0]
                radius = chosen[2]

                if prev_x[i] is not None:
                    dx = int(center_x) - int(prev_x[i])
                    dx_values[i].append(dx)
                    dx_values_since_change[i].append(dx)
                    if len(dx_values[i]) > FILTER_LENGTH:  # If the length exceeds the filter length, remove the oldest value
                        dx_values[i].pop(0)
                    avg_dx[i] = np.mean(dx_values[i])  # Calculate average dx

                    # Check if the circle changes direction
                    current_sign = np.sign(avg_dx[i])
                    if prev_sign[i] is not None and current_sign != prev_sign[i]:
                        peak_speed = max([abs(dx) for dx in dx_values_since_change[i]])
                        dx_values_since_change[i] = []  # Reset the dx_values_since_change
                        play_sound(i, current_sign, peak_speed)

                    prev_sign[i] = current_sign

                prev_x[i] = center_x

                color = (0, 255, 0) if i == 0 else (0, 0, 255)

                # Draw circle and center point on the frame
                cv.circle(frame, (chosen[0], chosen[1]), radius, color, 3)
                cv.circle(frame, (chosen[0], chosen[1]), 1, color, 3)

                cv.putText(frame, 'dx {}: {:.2f}'.format(i, avg_dx[i]), (10, VIDEO_HEIGHT - 20 - 20*i), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)

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