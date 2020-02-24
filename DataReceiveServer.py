import socket
import struct
import threading
import time
import datetime
import cv2
import sys
import time
import multiprocessing
import logging
import numpy as np
import RPi.GPIO as GPIO
from objrecog_test import CameraClient
## imports for object detection
import os, shutil, random
import matplotlib.pyplot as plt
import random
from slider import Slider
from binaryclassifier import BinaryClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import data, exposure
from joblib import dump, load
from Bluetooth_Connection import HxmClient
from gyro2 import GyroClient
from queue import Queue
import queue

class ReceiveServer:
    def __init__(self):
        self.t1 = threading.Thread(target = self.receivingThread1)
        self.t2 = threading.Thread(target = self.receivingThread2)
        self.t3 = threading.Thread(target = self.processingThread)
        self.qSize = 50
        self.recvQ = Queue()
        self.gyroQ = Queue()
        hrFileName = "HeartRate" + "_User_Log(" + str(datetime.date.today()) + " " + str(datetime.datetime.now().hour) + "." + str(datetime.datetime.now().minute) + "." + str(datetime.datetime.now().second)
        mpuFileName = "Gyro" + "_User_Log(" + str(datetime.date.today()) + " " + str(datetime.datetime.now().hour) + "." + str(datetime.datetime.now().minute) + "." + str(datetime.datetime.now().second)
        self.hrFile = open(hrFileName, "w")
        self.mpuFile = open(mpuFileName, "w")
        
    def saveToFile(self, fileName):
        fname = str(fileName) + "_User_Log(" + str(datetime.date.today()) + " " + str(datetime.datetime.now().hour) + "." + str(datetime.datetime.now().minute) + "." + str(datetime.datetime.now().second) + ").txt"
        fo = open(fname, "w")
        if(self.head == -1):
            fo.write("Queue is empty")
        elif (self.tail >= self.head):
            for i in range(self.head, self.tail+1):
                fo.write(str(self.queue[i])+"\n")
        else:
            for i in range(self.head, self.maxSize):
                fo.write(str(self.queue[i])+"\n")
            for i in range(0, self.tail+1):
                fo.write(str(self.queue[i])+"\n")
        fo.close()

    def processingThread(self):
        #time.sleep(5)
        logging.warning("Sensing ...")
        rates = []
        avgRate = 0
        prev_hr = 0
        while True:
            #print('here')
            try:
                #if(self.recvQ.get(block=False) != None):
                #   time.sleep(5)
                #print(self.recvQ.get(block = False))
                gyro = self.gyroQ.get(block = False)
                if gyro != None:
                    #logging.warning("HERE")
                    if(float(gyro[1]) >= 50 or float(gyro[1]) <= -50):
                        logging.warning("You are unbalanced!")
                        #set GPIO pin 17 to high to sound alarm in order to alert the user
                        GPIO.output(17, True)
                heart_rate = int(self.recvQ.get(block = False)[1])
                if(prev_hr != 0):
                    if(heart_rate - prev_hr >= 2):
                        logging.warning("Spike in heart rate Detected")
                        #set GPIO pin 17 to high to sound alarm in order to alert the user
                        GPIO.output(17, True)
        
                
                prev_hr = heart_rate
                
                
                #for i in range(0, 5):
                #    avgRate += int(self.recvQ.get(block = False)[1])
                #avgRate = avgRate / 5
                #logging.warning(avgRate)
            
            except queue.Empty:
                pass

    def receivingThread1(self):
        HOST = '127.0.0.1'
        PORT = 52222
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            while True:
                s.listen(1)
                conn, addr = s.accept()
                with conn:
                    received = conn.recv(512)
                    recvString = received.decode()
                    print(recvString)
                    recData = recvString.split(',')
                    self.hrFile.write(recvString + "\n")
                    self.hrFile.flush()
                    if self.recvQ.full():
                        self.recvQ.get()
                        self.recvQ.put(recData)
                        self.hrFile.close()
                        hrFileName = "HeartRate" + "_User_Log(" + str(datetime.date.today()) + " " + str(datetime.datetime.now().hour) + "." + str(datetime.datetime.now().minute) + "." + str(datetime.datetime.now().second)
                        self.hrFile = open(hrFileName, "w")
                    else:
                        self.recvQ.put(recData)
                    #rint(recData)

    def receivingThread2(self):
        """
        Ethernet connection with laptop
        """
        HOST = '127.0.0.1'
        PORT = 53333
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            while True:
                s.listen(1)
                conn, addr = s.accept()
                with conn:
                    received = conn.recv(512)
                    recString = received.decode()
                    recdata = recString.split(',')
                    self.mpuFile.write(recString + "\n")
                    self.mpuFile.flush()
                    if self.gyroQ.full():
                        self.gyroQ.get()
                        self.gyroQ.put(recData)
                        self.mpuFile.close()
                        mpuFileName = "mpu" + "_User_Log(" + str(datetime.date.today()) + " " + str(datetime.datetime.now().hour) + "." + str(datetime.datetime.now().minute) + "." + str(datetime.datetime.now().second)
                        self.mpuFile = open(mpuFileName, "w")
                    else:
                        self.gyroQ.put(recdata)
                    #print(recdata)

    def runServer(self):
        self.t1.start()
        self.t2.start()
        self.t3.start()

def sliderThread(name, frame, boxes):
    logging.info("Thread %s : starting...", name)
    for (x, y, window) in Slider(frame, name, stepSize=16, windowSize = (64, 64)):# name is a numerical value that determines what row this thread should handle
        ex_features, ex_hog_image = hog(window, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
        arr = ex_features.reshape(1, -1)
        if(arr.shape == (1, 128)):
            if bc.predict(arr):
                boxes.append((x, y, window))


if __name__ == "__main__":
    initTime = time.time()
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(21, GPIO.OUT)
    GPIO.setup(17, GPIO.OUT)
    server = ReceiveServer()
    server.runServer()
    hxmClient = HxmClient(addr=('127.0.0.1', 52222), initTime=initTime)
    hxmClient.start()
    gyroClient = GyroClient(addr=('127.0.0.1', 53333), initTime=initTime)
    gyroClient.start()
   
    ### find accuracy with test data ###

    """
    TN = 0
    TP = 0
    FP = 0
    FN = 0

    tp_test = time.time()
    bc = BinaryClassifier(svc, scaler)
    
    if os.path.exists(pos_test):
        posFiles = os.listdir(pos_test)
        for img in posFiles:
            image = cv2.imread(pos_test + "/" + img, cv2.COLOR_BGR2GRAY)
            features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
            arr = features.reshape(1, -1)
            if bc.predict(arr):
                TP = TP + 1
            else:
                FN = FN + 1
    if os.path.exists(neg_test):
        negFiles = os.listdir(neg_test)
        for img in negFiles:
            image = cv2.imread(neg_test+"/"+img, cv2.COLOR_BGR2GRAY)
            features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
            arr = features.reshape(1, -1)
            if bc.predict(arr):
                FP = FP + 1
            else:
                TN = TN + 1
    posFiles = os.listdir(pos_test)
    negFiles = os.listdir(neg_test)
    posTotal = int(len(posFiles))
    negTotal = int(len(negFiles))
    print("True positive percentage: ", str((TP/posTotal)*100))
    print("True negative percentage: ", str((TN/negTotal)*100))
    print("False positive percentage: ", str((FP/negTotal)*100))
    print("False negative percentage: ", str((FN/posTotal)*100))
    #end of training
    """
    #start clients
    cameraClient = CameraClient()
    cameraClient.start()

    # logging configuration
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt = "%H:%M:%S")

    (winW, winH) = (64, 64)
    boxes = []
    
    pos_test = "./test/pos"
    neg_test = "./test/neg"

    svc = load('svc.joblib')
    scaler = load('scaler.joblib')

    bc = BinaryClassifier(svc, scaler)
    ### Signal that program is ready to receive images
    cap = cv2.VideoCapture(0) ## if using live image stream
    print("entering while loop...")

    while True:
        try:
            # wait for program to send image
            # once image is received:
            output = []
            frame = cameraClient.frameQueue.get()
                
            # use the slider to feed the classifier the window
            # define window width and height

            print("Starting detection...")
            timeDetect = time.time()
            x = threading.Thread(target = sliderThread, args=(0, frame, boxes,), daemon=True)
            x1 = threading.Thread(target = sliderThread, args=(1, frame, boxes,), daemon=True)
            x2 = threading.Thread(target = sliderThread, args=(2, frame, boxes,), daemon=True)
            x3 = threading.Thread(target = sliderThread, args=(3, frame, boxes,), daemon=True)
            x4 = threading.Thread(target = sliderThread, args=(4, frame, boxes,), daemon=True)
            x5 = threading.Thread(target = sliderThread, args=(5, frame, boxes,), daemon=True)
            x6 = threading.Thread(target = sliderThread, args=(6, frame, boxes,), daemon=True)
            x7 = threading.Thread(target = sliderThread, args=(7, frame, boxes,), daemon=True)
            #x8 = threading.Thread(target = sliderThread, args=(8, frame, boxes,), daemon=True)
            #x9 = threading.Thread(target = sliderThread, args=(9, frame, boxes,), daemon=True)
            #x10 = threading.Thread(target = sliderThread, args=(10, frame, boxes,), daemon=True)
            #x11 = threading.Thread(target = sliderThread, args=(11, frame, boxes,), daemon=True)
            logging.info("Main: before starting threads...")
            x.start()
            x1.start()
            x2.start()
            x3.start()
            x4.start()
            x5.start()
            x6.start()
            x7.start()
            #x8.start()
            #x9.start()
            #x10.start()
            #x11.start()
            logging.info("Main: wait for threads to finish...")
            x.join()
            x1.join()
            x2.join()
            x3.join()
            x4.join()
            x5.join()
            x6.join()
            x7.join()
            #x8.join()
            #x9.join()
            #x10.join()
            #x11.join()
            logging.info("Main: threads are done")
                        
            print("size of boxes: ", len(boxes))
            print("Time to detection: ", np.round(time.time()-timeDetect, 2))
            # draw boxes from the array over the image
            #for (x, y, window) in boxes:  
                #label = "{:.2f}%".format(conf*100)
            #    cv2.rectangle(frame, (x,y),(x+winW, y+winH), (0,255,255), 2)
                #cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                    
            #cv2.imshow('Camera Frame', frame)

            # send 'image' to server
            #key=cv2.waitKey(0)
            boxes.clear()
            GPIO.output(21, True)
            time.sleep(0.5)
            GPIO.output(21, False)
            time.sleep(0.5)
            #if (key==27): # Esc key
            #    self.Re
            #    break
            
            #cap.release()
            #cv2.destroyAllWindows()
                
        except Exception as e:
            logging.error("Problem getting frames!!!")
            cameraClient.stop_Queue.cancel_join_thread()
            cameraClient.stop_Queue.close()
            cameraClient.frameQueue.close()
            cameraClient.join(10)
            raise e
            #
        pass
