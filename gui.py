import numpy as np
import sys

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import logging as log
import argparse
import time

sys.path.append(
    'C:\\Program Files (x86)\\Intel\\openvino_2021.4.582\\deployment_tools\\open_model_zoo\\demos\\common\\python')
from images_capture import open_images_capture


class Gui:
    def __init__(self, input_cap, camera=True):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        if camera:
            self.vs = cv2.VideoCapture(0)
            self.camera = True
        else:
            self.vs = open_images_capture(input_cap, True)
            self.camera = False

        self.current_image = None  # current image from the camera

        self.root = tk.Tk()  # initialize root window
        self.root.title("no-mask-no-service")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)  # initialize image panel
        self.panel.pack(side=tk.LEFT)

        # create a button, that when pressed, will take the current frame and save it to file
        # btn = tk.Button(self.root, text="Snapshot!", command=self.take_snapshot)
        # btn.pack(fill="both", expand=True, padx=10, pady=10)

        # Slider window (slider controls stage position)
        self.sliderFrame = tk.Frame(self.root, width=600, height=150)
        self.sliderFrame.pack(padx=10, pady=2)

        self.networkLabel = tk.Label(self.sliderFrame, text='network mode')
        self.mainMode = ttk.Combobox(self.sliderFrame,
                                     values=[
                                         "Aizoo",
                                         "face detector",
                                         "detector + Aizoo"])
        self.reLabel = tk.Label(self.sliderFrame, text='re-identidication mode')
        self.re_identificationMode = ttk.Combobox(self.sliderFrame,
                                                  values=[
                                                      "re-identification",
                                                      "no re-identification"])
        self.deviceLabel = tk.Label(self.sliderFrame, text='device mode')
        self.deviceMode = ttk.Combobox(self.sliderFrame,
                                       values=[
                                           "CPU",
                                           "GPU",
                                           "HETERO"])

        self.atrashLabel = tk.Label(self.sliderFrame, text='Aizoo threshold')
        self.aizooThreshold = tk.Entry(self.sliderFrame, width=23, text='Aizoo threshold')
        self.dtrashLabel = tk.Label(self.sliderFrame, text='detector threshold')
        self.detectorThreshold = tk.Entry(self.sliderFrame, width=23, text='detector threshold')
        self.rtrashLabel = tk.Label(self.sliderFrame, text='re-identification threshold')
        self.re_identificationThreshold = tk.Entry(self.sliderFrame, width=23, text='re-identification threshold')

        self.videoPathLabel = tk.Label(self.sliderFrame, text='full path to video')
        self.videoPathEntry = tk.Entry(self.sliderFrame, width=23, text='full path to video')

        self.startButton = tk.Button(self.sliderFrame, bg="red", fg="blue", text="Start")

        self.startButton.bind('<ButtonRelease-1>', self.configure_and_start_processing())

        self.networkLabel.pack()
        self.mainMode.pack()
        self.reLabel.pack()
        self.re_identificationMode.pack()
        self.deviceLabel.pack()
        self.deviceMode.pack()

        self.atrashLabel.pack()
        self.aizooThreshold.pack()
        self.dtrashLabel.pack()
        self.detectorThreshold.pack()
        self.rtrashLabel.pack()
        self.re_identificationThreshold.pack()
        
        self.videoPathLabel.pack()
        self.videoPathEntry.pack()

        self.startButton.pack()

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame

        # self.video_loop()

    def configure_and_start_processing(self):
        # TODO
        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        start_time = time.time()  # start time of the loop
        ok, frame = 0, 0
        if self.camera:
            ok, frame = self.vs.read()  # read frame from video stream
            frame = cv2.flip(frame, 1)
        else:
            frame = self.vs.read()
            ok = True

        if ok:  # frame captured without any errors
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)  # show the image
        print("FPS: ", 1.0 / (time.time() - start_time))
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds

    # def take_snapshot(self):
    #     """ Take snapshot and save it to the file """
    #     ts = datetime.datetime.now()  # grab the current timestamp
    #     filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
    #     p = os.path.join(self.output_path, filename)  # construct output path
    #     self.current_image.save(p, "JPEG")  # save image as jpeg file
    #     print("[INFO] saved {}".format(filename))

    def destructor(self):
        """ Destroy the root object and release all resources """
        # print("[INFO] closing...")
        log.info("closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", default="./",
#                 help="path to output directory to store snapshots (default: current folder")
# args = vars(ap.parse_args())


log.basicConfig(format="[ %(levelname)s ] %(message)s",
                level=log.INFO, stream=sys.stdout)
# start the app
log.info("starting...")
pba = Gui("videocam\\videcam2.mov", camera=False)
pba.root.mainloop()
