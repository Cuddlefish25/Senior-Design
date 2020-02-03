import cv2
import sys
import time
import multiprocessing
import logging
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np

class getFrames(multiprocessing.Process):
    def __init__(self):
        self.frameQueue = multiprocessing.Queue(maxsize=1)
        self.stop_Queue = multiprocessing.JoinableQueue(maxsize=1)
        self.target_fps = 20
        self.target_res = (640,480)
        super(getFrames, self).__init__()
        
    def run(self):
        logging.warning('start')
        try:
            with PiCamera() as capture:
                logging.warning(capture)
                capture.resolution = self.target_res
                capture.framerate = self.target_fps
                raw_capture = PiRGBArray(capture, size=self.target_res)
                for frame in capture.capture_continuous(raw_capture, format="bgr", use_video_port=True):
                    if not self.stop_Queue.empty():
                        logging.warning("The stop queue is not empty. Stop acquiring frames.")
                        self.stop_Queue.get()
                        self.stop_Queue.task_done()
                        break
                    raw_capture.truncate(0)
                    out = cv2.cvtColor(frame.array,cv2.COLOR_BGR2GRAY)
                    self.frameQueue.put(out)
        finally:
            logging.warning("Closing frame grabber process.")
            self.stop_Queue.close()
            self.frameQueue.close()
            logging.warning("Camera Frame grabber stopped acquisition successfully.")
         
if __name__ == "__main__":
    framegrabber = getFrames()
    framegrabber.start()
    while(True):
        try:
            img = framegrabber.frameQueue.get(timeout=10)
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            lower_blue = np.array([110,50,50])
            upper_blue = np.array([130,255,255])
            mask = cv2.inRange(frame, lower_blue, upper_blue)
            cv2.imshow('mask',mask)
            #cv2.imshow("test", frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
                logging.warning("esc hit")
                framegrabber.stop_Queue.put(1)
        except Exception as e:
            logging.error("Problem getting frames!!!")
            framegrabber.stop_Queue.cancel_join_thread()
            framegrabber.stop_Queue.close()
            framegrabber.frameQueue.close()
            framegrabber.join(10)
            raise e
        
                    