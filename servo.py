import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(32, GPIO.OUT)
pwm = GPIO.PWM(32, 50)
pwm.start(0)

def SetAngle(angle):
	duty = angle / 18 + 2
	GPIO.output(32, GPIO.HIGH)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(32, GPIO.LOW)
	pwm.ChangeDutyCycle(0)

while True:
    SetAngle(90)
    print("90")
    sleep(2)
    SetAngle(40)
    print("40")
    sleep(1)
    SetAngle(130)
    print("140")
    sleep(1)