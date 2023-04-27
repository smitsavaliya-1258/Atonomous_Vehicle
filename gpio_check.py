import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Blink
pin = 38
pin2 = 40 # if none set to 40
GPIO.setup(pin,GPIO.OUT)
GPIO.setup(pin2,GPIO.OUT)

# Distance
GPIO_TRIGGER = 16
GPIO_ECHO = 12
vcc = 36
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(vcc, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
GPIO.output(GPIO_TRIGGER, GPIO.LOW)
time.sleep(2)

def distance():
    GPIO.setup(vcc, GPIO.HIGH)
    GPIO.output(GPIO_TRIGGER, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, GPIO.LOW)
    StartTime = time.time()
    StopTime = time.time()

    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()

    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()

    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    distance = (TimeElapsed * 17150, 2)
    print("Measured Distance = ", distance, "cm")
    return distance

def blink():
    GPIO.output(pin2,GPIO.HIGH)
    GPIO.output(pin,GPIO.HIGH)
    print("On")
    time.sleep(1)
    GPIO.output(pin,GPIO.LOW)
    print("Off")
    time.sleep(1)
    return

while True:
    distance()
    #blink()