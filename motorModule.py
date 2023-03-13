import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor():
    def __init__(self,M1A,M1B,M2A,M2B):
        self.m1a = M1A
        self.m1b = M1B
        self.m2a = M2A
        self.m2b = M2B
        GPIO.setup(self.m1a,GPIO.OUT);GPIO.setup(self.m1b,GPIO.OUT)
        GPIO.setup(self.m2a,GPIO.OUT);GPIO.setup(self.m2b,GPIO.OUT)
        
        self.pwm_m1a = GPIO.PWM(self.m1a, 100)
        self.pwm_m1b = GPIO.PWM(self.m1b, 100)
        self.pwm_m2a = GPIO.PWM(self.m2a, 100)
        self.pwm_m2b = GPIO.PWM(self.m2b, 100)
        self.pwm_m1a.start(0)
        self.pwm_m1b.start(0)
        self.pwm_m2a.start(0)
        self.pwm_m2b.start(0)
        self.mySpeed=0

    def m_c(self,a2,b2,a1,b1):
        self.pwm_m1a.start(a1)
        self.pwm_m1b.start(b1)
        self.pwm_m2a.start(a2)
        self.pwm_m2b.start(b2)

    def move(self,speed=0.5,turn=0,t=0):
        speed *=100
        speed = speed//2
        turn *=70
        leftSpeed = speed-turn
        rightSpeed = speed+turn
 
        if leftSpeed>100: leftSpeed =100
        elif leftSpeed<-100: leftSpeed = -100
        if rightSpeed>100: rightSpeed =100
        elif rightSpeed<-100: rightSpeed = -100
        # print(leftSpeed,rightSpeed)
        if leftSpeed>0:
            if rightSpeed>0:
                self.m_c(abs(leftSpeed),0,abs(rightSpeed),0)
            else:
                self.m_c(abs(leftSpeed),0,0,abs(rightSpeed))
        else:
            if rightSpeed>0:
                self.m_c(0,abs(leftSpeed),abs(rightSpeed),0)
            else:
                self.m_c(0,abs(leftSpeed),0,abs(rightSpeed))
        sleep(t)
    def drive(self, l_spd=20, r_spd=20, t=0.05):
        if (l_spd>=0):
            if(r_spd>=0):
                self.m_c(abs(l_spd), 0, abs(r_spd), 0)
            else:
                self.m_c(abs(l_spd), 0, 0, abs(r_spd))
        else:
            if(r_spd>=0):
                self.m_c(0, abs(l_spd), abs(r_spd), 0)
            else:
                self.m_c(0, abs(l_spd), 0, abs(r_spd))
        sleep(t)
 
    def stop(self,t=0):
        self.pwm_m1a.ChangeDutyCycle(0)
        self.pwm_m1b.ChangeDutyCycle(0)
        self.pwm_m2a.ChangeDutyCycle(0)
        self.pwm_m2b.ChangeDutyCycle(0)
        self.mySpeed=0
        sleep(t)
 
# def main():
#     motor.move(0.2,0,2)
#     motor.stop(1)
#     motor.move(-0.2,0,2)
#     motor.stop(1)
#     motor.move(0.2,0.2,2)
#     motor.stop(1)
#     motor.move(0.2,-0.2,2)
#     motor.stop(1)
 
# if __name__ == '__main__':
#     motor= Motor(17,27,10,9)
#     main()