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
from Bluetooth_Connection import HxmClient
from gyro2 import GyroClient
from objrecog_test import CameraClient
import queue
from queue import Queue

class CircularBuffer:

    def __init__(self, maxSize = 3600):
        self.queue = [None for i in range(maxSize)]
        self.head = -1
        self.tail = -1
        self.maxSize = maxSize

    def push(self, data):
        if self.isFull():
            print("Queue Full")
            #file output when queue full and flush one value
            return False
        if self.isEmpty():
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = data
        else:
            self.tail = (self.tail+1)%self.maxSize
            self.queue[self.tail] = data
        return True

    def pop(self):
        if self.isEmpty():
            print("Queue Empty")
            #not allowed!!
        elif self.head == self.tail:
            data = self.queue[self.head]
            self.head = -1
            self.tail = -1
            return data
        else:
            data = self.queue[self.head]
            self.head = (self.head + 1) % self.maxSize
            return data

    def isFull(self):
        return ((self.tail+1) % self.maxSize == self.head)

    def isEmpty(self):
        return (self.head == -1)

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

class ReceiveServer:
    def __init__(self):
        self.t1 = threading.Thread(target = self.receivingThread1)
        self.t2 = threading.Thread(target = self.receivingThread2)
        self.t3 = threading.Thread(target = self.processingThread)
        #self.recvQ = []
        self.qSize = 50
        self.recvQ = Queue()
        #self.gyroQ = CircularBuffer(maxSize=self.qSize)
        self.gyroQ = Queue()

    def processingThread(self):
        #time.sleep(5)
    
        rates = []
        avgRate = 0
        while True:
            #print('here')
            try:
                if(self.recvQ.get(block=False) != None):
                   time.sleep(5)
                #print(self.recvQ.get(block = False))
                #gyro = self.gyroQ.get(block = False)
                #if gyro != None:
                #    if(float(gyro[1]) >= 100 or float(gyro[1]) <= -100):
                #        logging.warning("You are unbalanced!")
                for i in range(0, 5):
                    avgRate += int(self.recvQ.get(block = False)[1])
                avgRate = avgRate / 5
                logging.warning(avgRate)
            
            except queue.Empty:
                pass
    
        
        

    def receivingThread1(self):
        HOST = '127.0.0.1'
        PORT = 52222
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            temp = 0
            while True:
                s.listen(1)
                conn, addr = s.accept()
                with conn:
                    received = conn.recv(512)
                    recData = received.decode().split(',')
                    if (temp==self.qSize):
                        print("heart rate queue is full write to file")
                        self.recvQ.saveToFile("HeartRate")
                        temp = 0
                    #if self.recvQ.isFull():
                    #    self.recvQ.get()
                    self.recvQ.put(recData)
                    temp+=1
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
            temp = 0
            while True:
                s.listen(1)
                conn, addr = s.accept()
                with conn:
                    #print("Connected!")
                    received = conn.recv(512)
                    recdata = received.decode().split(',')
                    #if (temp==self.qSize):
                        #print("gyro queue is full write to file")
                        #self.gyroQ.saveToFile("Gyro")
                        #temp = 0
                    #if self.gyroQ.isFull():
                    #    self.gyroQ.pop()
                    self.gyroQ.put(recdata)
                    #temp+=1
                    #print(recdata)

    def runServer(self):
        self.t1.start()
        self.t2.start()
        self.t3.start()

if __name__ == "__main__":
    initTime = time.time()
    server = ReceiveServer()
    server.runServer()
    hxmClient = HxmClient(addr=('127.0.0.1', 52222), initTime=initTime)
    hxmClient.start()
    gyroClient = GyroClient(addr=('127.0.0.1', 53333), initTime=initTime)
    gyroClient.start()
    #cameraClient = CameraClient()
    #cameraClient.start()
    while True:
        """
        try:
            img = cameraClient.frameQueue.get(timeout=10)
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            lower_blue = np.array([110,50,50])
            upper_blue = np.array([130,255,255])
            mask = cv2.inRange(frame, lower_blue, upper_blue)
            logging.warning(time.time()-initTime)
            cv2.imshow('mask',mask)
            #cv2.imshow("test", frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
                logging.warning("esc hit")
                cameraClient.stop_Queue.put(1)
        except Exception as e:
            logging.error("Problem getting frames!!!")
            cameraClient.stop_Queue.cancel_join_thread()
            cameraClient.stop_Queue.close()
            cameraClient.frameQueue.close()
            cameraClient.join(10)
            raise e
        """
        #
        pass
