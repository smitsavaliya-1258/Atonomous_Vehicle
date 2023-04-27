import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import math

cap = cv2.VideoCapture(0)  #for webcam
#cap = cv2.VideoCapture("data/road.mp4")

STERRING_PIN = 32
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(STERRING_PIN, GPIO.OUT)
pwm = GPIO.PWM(STERRING_PIN, 50)
pwm.start(0)

def SetAngle(angle):
	duty = angle / 18 + 2
	GPIO.output(STERRING_PIN, True)
	pwm.ChangeDutyCycle(duty)
	time.sleep(1)
	GPIO.output(STERRING_PIN, False)
	pwm.ChangeDutyCycle(0)



height = 480
width = 640
cap.set(3, width)
cap.set(4, height)
state = True
lane_lines = []
left_fit = []
right_fit = []

boundary = 1 / 3
left_region_boundary = height * (1 - boundary)
right_region_boundary = height * boundary

def detect_edges(frame):
    edges = cv2.Canny(frame, 50, 100)
    return edges

def region_of_interest(edges):
    pts1 = np.float32([[0, 300], [350, 300], [0, 470], [350, 470]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    cropped_edges = cv2.warpPerspective(frame, M, (320, 480))
    cv2.imshow("roi", cropped_edges)
    return cropped_edges

def detect_line_segments(cropped_edges):
    line_segments = cv2.HoughLinesP(edge,
                            # smaller rho/theta=more accurate longer processing time
                            rho=6,  # number of pixels
                            theta=np.pi / 60,
                            threshold=200,  #default 160
                            lines=np.array([]),
                            minLineLength=20, #default 10
                            maxLineGap=10)  #default 10
    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        print("")
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
                #print("Skipping:- slope = infinity")
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
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

    x1 = int(height/3)
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
    print(steering_angle)
    # 0-left, 300-right, 133-straight
    if steering_angle in range(40, 140):
        SetAngle(steering_angle)

    return steering_angle


while True:
    ret, img = cap.read()
    #img = cv2.resize(img, (width, height))
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(blur, 50, 100)  # default 50 , 100

    line_segments = detect_line_segments(frame)
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    heading_image = display_heading_line(frame, steering_angle)

    cv2.imshow("Raw Video", frame)
    # cv2.imshow("Region Of Interest", edge)
    cv2.imshow("Detected lines", lane_lines_image)
    cv2.imshow("Steering Direction line", heading_image)


    if cv2.waitKey(1) and 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break