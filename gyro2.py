from mpu6050 import mpu6050
import time


sensor = mpu6050(0x68) #create instance of MPU6050 unit
                       #(at address 0x68 on I2C interface)
def getValues():
    while(True):
        accel_data = sensor.get_accel_data() #read acceleration data, get method returns a dictionary
        print("Raw Acceleration Data:")   #print the raw data in real time
        print("X: ", accel_data.get('x'))
        print("Y: ", accel_data.get('y'))
        print("Z: ", accel_data.get('z'))
        gyro_data = sensor.get_gyro_data() #read gyroscope data, get method returns a dictionary
        print("Raw Gyroscope Data:")    #print the raw data in real time
        print("X: ", gyro_data.get('x'))
        print("Y: ", gyro_data.get('y'))
        print("Z: ", gyro_data.get('z'))
        time.sleep(1) #delay for 1 sec before next read

def main():
    try:
        getValues()
    except KeyboardInterrupt: #CTRL-C to exit
            pass
main()
