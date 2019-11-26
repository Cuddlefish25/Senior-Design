from mpu6050 import mpu6050
import time


sensor = mpu6050(0x68)
def getValues():
    while(True):
        accel_data = sensor.get_accel_data()
        print("Raw Acceleration Data:")
        print("X: ", accel_data.get('x'))
        print("Y: ", accel_data.get('y'))
        print("Z: ", accel_data.get('z'))
        gyro_data = sensor.get_gyro_data()
        print("Raw Gyroscope Data:")
        print("X: ", gyro_data.get('x'))
        print("Y: ", gyro_data.get('y'))
        print("Z: ", gyro_data.get('z'))
        time.sleep(1)

def main():
    try:
        getValues()
    except KeyboardInterrupt:
            pass
main()
