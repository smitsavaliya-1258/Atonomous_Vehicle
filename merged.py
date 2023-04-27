print("Loading Library, Models & Initializing Camera, Wait...")
import cv2
import time
import math
import numpy as np
import RPi.GPIO as GPIO
from keras.models import load_model

ENR = 11
FORWARD_PIN_R = 15
REVERSE_PIN_R = 13

ENL = 19
FORWARD_PIN_L = 21
REVERSE_PIN_L = 23

STERRING_PIN = 32
BRAKE_LIGHT_R = 37
BRAKE_LIGHT_L = 35

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

GPIO.setup(ENR, GPIO.OUT)
GPIO.setup(FORWARD_PIN_R, GPIO.OUT)
GPIO.setup(REVERSE_PIN_R, GPIO.OUT)
GPIO.setup(ENL, GPIO.OUT)
GPIO.setup(FORWARD_PIN_L, GPIO.OUT)
GPIO.setup(REVERSE_PIN_L, GPIO.OUT)

GPIO.setup(STERRING_PIN, GPIO.OUT)
pwm = GPIO.PWM(STERRING_PIN, 50)
pwm.start(0)

GPIO.setup(BRAKE_LIGHT_R, GPIO.OUT)
GPIO.setup(BRAKE_LIGHT_L, GPIO.OUT)
GPIO.output(BRAKE_LIGHT_R, GPIO.LOW)
GPIO.output(BRAKE_LIGHT_L, GPIO.LOW)

cap = cv2.VideoCapture(0)
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def forward():
    GPIO.output(ENR, GPIO.HIGH)
    GPIO.output(FORWARD_PIN_R, GPIO.HIGH)
    GPIO.output(REVERSE_PIN_R, GPIO.LOW)
    GPIO.output(ENL, GPIO.HIGH)
    GPIO.output(FORWARD_PIN_L, GPIO.HIGH)
    GPIO.output(REVERSE_PIN_L, GPIO.LOW)
    GPIO.output(BRAKE_LIGHT_R, GPIO.LOW)
    GPIO.output(BRAKE_LIGHT_L, GPIO.LOW)
    return

def reverse():
    GPIO.output(ENR, GPIO.HIGH)
    GPIO.output(FORWARD_PIN_R, GPIO.LOW)
    GPIO.output(REVERSE_PIN_R, GPIO.HIGH)
    GPIO.output(ENL, GPIO.HIGH)
    GPIO.output(FORWARD_PIN_L, GPIO.LOW)
    GPIO.output(REVERSE_PIN_L, GPIO.HIGH)
    return

def stop():
    GPIO.setwarnings(False)
    GPIO.output(ENR, GPIO.LOW)
    GPIO.output(FORWARD_PIN_R, GPIO.LOW)
    GPIO.output(REVERSE_PIN_R, GPIO.LOW)
    GPIO.output(ENL, GPIO.LOW)
    GPIO.output(FORWARD_PIN_L, GPIO.LOW)
    GPIO.output(REVERSE_PIN_L, GPIO.LOW)
    GPIO.output(BRAKE_LIGHT_R, GPIO.HIGH)
    GPIO.output(BRAKE_LIGHT_L, GPIO.HIGH)
    return

def SetAngle(angle):
	duty = angle / 18 + 2
	GPIO.output(STERRING_PIN, True)
	pwm.ChangeDutyCycle(duty)
	time.sleep(1)
	GPIO.output(STERRING_PIN, False)
	pwm.ChangeDutyCycle(0)

def detect_edges(frame):
    edges = cv2.Canny(frame, 50, 100)
    return edges

def region_of_interest(edges):
    pts1 = np.float32([[0, 300], [350, 300], [0, 470], [350, 470]])
    pts2 = np.float32([[0, 0], [224, 0], [0, 224], [224, 224]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    cropped_edges = cv2.warpPerspective(image, M, (224, 224))
    return cropped_edges

def detect_line_segments(cropped_edges):
    line_segments = cv2.HoughLinesP(edge,
                                    # smaller rho/theta=more accurate longer processing time
                                    rho=6,
                                    theta=np.pi / 60,
                                    threshold=120,  # default 100
                                    lines=np.array([]),
                                    minLineLength=20,  # default 20
                                    maxLineGap=5)  # default 5
    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        print("Put Car on the Road ( No Road Lane Lines Found !)")
        stop()
        return lane_lines

    width, height, _ = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1 / 3
    left_region_boundary = height * (1 - boundary)
    right_region_boundary = height * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                # print("Skipping:- slope = infinity")
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
    return lane_lines

def make_points(frame, line):
    width, height, _ = frame.shape
    slope, intercept = line
    y1 = width  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0:
        slope = 0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(height / 2)
    y1 = height
    x2 = int(x1 - width / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image

def get_steering_angle(frame, lane_lines):
    width, height, _ = frame.shape

    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(height / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(width / 2)

    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(width / 2)

    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(width / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90
    #print(steering_angle)

    # 40-left, 140-right, 90-straight
    if steering_angle in range(40, 140):
        SetAngle(steering_angle)
        forward()

    return steering_angle

while True:
    start_time = time.time()
    ret, image = cap.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    frame = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur1 = cv2.GaussianBlur(gray, (3, 3), 3)
    blur = cv2.GaussianBlur(blur1, (3, 3), 0)
    edge = cv2.Canny(blur, 30, 150)  # best values 30 , 150

    line_segments = detect_line_segments(image)
    lane_lines = average_slope_intercept(image, line_segments)

    lane_lines_image = display_lines(image, lane_lines)
    steering_angle = get_steering_angle(image, lane_lines)
    heading_image = display_heading_line(image, steering_angle)

    #cv2.imshow("Webcam Image", image)  # for techable machine

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    confidence_score = str(np.round(confidence_score * 100))
    FRAME = cv2.putText(heading_image, class_name[:], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    end_time = time.time()
    total_time = str(round(((end_time - start_time) * 10)+1.3,1))
    frame = cv2.putText(frame, "Prediction: ", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, confidence_score, (120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "%", (185, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Class: ", (2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, class_name[2:-1], (70, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Delay: ", (2, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, total_time, (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Sec", (125, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


    # print("index", index)
    #cv2.imshow("Edges", edge)
    #cv2.imshow("lane Lines", lane_lines_image)
    #cv2.imshow("Final", FRAME)
    cv2.imshow("frame Image", frame)


    if cv2.waitKey(1) and 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break