import RPi.GPIO as GPIO
import time

try:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO_TRIG = 16
    GPIO_ECHO = 12
    vcc = 36
    GPIO.setup(vcc, GPIO.OUT)
    GPIO.output(vcc, GPIO.HIGH)

    while True:
        GPIO.setup(GPIO_TRIG, GPIO.OUT)
        GPIO.setup(GPIO_ECHO, GPIO.IN)

        GPIO.output(GPIO_TRIG, False)
        time.sleep(0.2)
        GPIO.output(GPIO_TRIG, True)
        time.sleep(0.00001)
        GPIO.output(GPIO_TRIG, False)

        while GPIO.input(GPIO_ECHO)==0:
            start_time = time.time()

        while GPIO.input(GPIO_ECHO)==1:
            end_time = time.time()

        pulse_duration = end_time - start_time
        distance = pulse_duration * 17000
        distance = round(distance,2)
        print ("Distance: ", distance, "cm")
except:
    print("Code Stop Working")
    GPIO.cleanup()