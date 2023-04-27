print("Loading Library, Models & Initializing Camera, Wait...")
import cv2
import time
import numpy as np
import RPi.GPIO as GPIO
from keras.models import load_model

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
camera = cv2.VideoCapture(0)

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

while True:
    start_time = time.time()
    ret, image = camera.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    frame = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imshow("Webcam Image", image)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    confidence_score = str(np.round(confidence_score * 100))

    if index == 0:
        forward()
        SetAngle(90)
    elif index == 1:
        stop()
        SetAngle(90)
    elif index == 2:
        forward()
        SetAngle(115)
    elif index == 3:
        forward()
        SetAngle(65)
    elif index == 4:
        forward()
        SetAngle(40)
    elif index == 5:
        forward()
        SetAngle(140)

    end_time = time.time()
    total_time = str(round(((end_time - start_time) * 10)+1.3, 1))
    frame = cv2.putText(frame, "Prediction: ", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, confidence_score, (120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "%", (185, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Class: ", (2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, class_name[2:-1], (70, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Delay: ", (2, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, total_time, (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Sec", (125, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("frame Image", frame)

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()