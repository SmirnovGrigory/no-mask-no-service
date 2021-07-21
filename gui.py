import numpy as np
import sys
import imageio

import cv2
import tkinter as tk, threading
from tkinter import ttk
from PIL import Image, ImageTk

import logging as log
import argparse
import time

sys.path.append('C:\\Program Files (x86)\\Intel\\openvino_2021.4.582\\deployment_tools\\open_model_zoo\\demos\\common\\python')
from images_capture import open_images_capture
from main import gui_api


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
        self.root.geometry("1152x670")
        self.root.configure(bg="#E5E1D8")

        canvas = tk.Canvas(
            self.root,
            bg="#E5E1D8",
            height=700,
            width=1152,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        canvas.place(x=0, y=0)

        canvas.create_rectangle(
            0.0,
            0.0,
            815.0,
            700.0,
            fill="#C4C4C4",
            outline="")

        canvas.create_rectangle(
            0.0,
            500.0,
            1152.0,
            700.0,
            fill="#E5E5E5",
            outline="")

        canvas.create_text(
            580.0,
            534.0,
            text="Current parameters",
            fill="#645F5F",
            font=('Georgia', int(40.0), 'bold')
        )

        canvas.create_text(
            390.0,
            34.0,
            text="Output screen",
            fill="#645F5F",
            font=('Georgia', int(40.0), 'bold')
        )

        canvas.create_text(
            990.0,
            34.0,
            text="Settings",
            fill="#645F5F",
            font=('Georgia', int(40.0), 'bold')
        )

        canvas.create_rectangle(
            745.0,
            570.0,
            1017.0,
            650.0,
            fill="#C4C4C4",
            outline="")

        canvas.create_rectangle(
            440.0,
            570.0,
            712.0,
            650.0,
            fill="#C4C4C4",
            outline="")

        canvas.create_rectangle(
            135.0,
            570.0,
            407.0,
            650.0,
            fill="#C4C4C4",
            outline="")

        canvas.create_text(
            159.0,
            580.0,
            text="FPS:",
            fill="#645F5F",
            font=('Georgia', int(16.0), 'bold')
        )

        canvas.create_text(
            545.0,
            580.0,
            text="People With Masks:",
            fill="#645F5F",
            font=('Georgia', int(16.0), 'bold')
        )

        canvas.create_text(
            870.0,
            580.0,
            text="People Without Masks:",
            fill="#645F5F",
            font=('Georgia', int(16.0), 'bold')
        )

        self.panel = tk.Label(self.root)  # initialize image panel
        self.panel.pack(side=tk.LEFT)

        # Slider window (slider controls stage position)
        self.sliderFrame = tk.Frame(self.root, width=600, height=150, highlightthickness=1, highlightbackground="#C4C4C4")
        self.sliderFrame.place(in_=self.root, anchor="c", relx=.855, rely=.39)

        self.inputLabel = tk.Label(self.sliderFrame, text='Input Mode')
        self.inputMode = ttk.Combobox(self.sliderFrame,
                                     values=[
                                         "Image",
                                         "Video",
                                         "Web camera"])
        self.inputMode.set("Image")

        self.networkLabel = tk.Label(self.sliderFrame, text='Network Mode')
        self.mainMode = ttk.Combobox(self.sliderFrame,
                                     values=[
                                         "AIZOO",
                                         "Face Detector",
                                         "Face Detector + AIZOO"])
        self.mainMode.set("AIZOO")
        self.reLabel = tk.Label(self.sliderFrame, text='Re-Identification Mode')
        self.re_identificationMode = ttk.Combobox(self.sliderFrame,
                                                  values=[
                                                      "Re-Identification On",
                                                      "Re-Identification Off"])
        self.re_identificationMode.set("Re-Identification On")
        self.deviceLabel = tk.Label(self.sliderFrame, text='Device Mode')
        self.deviceMode = ttk.Combobox(self.sliderFrame,
                                       values=[
                                           "CPU",
                                           "GPU",
                                           "HETERO"])
        self.deviceMode.set("CPU")

        self.atrashLabel = tk.Label(self.sliderFrame, text='AIZOO threshold')
        self.aizooThreshold = tk.Entry(self.sliderFrame, width=23, text='AIZOO threshold')
        self.dtrashLabel = tk.Label(self.sliderFrame, text='Face Detector threshold')
        self.detectorThreshold = tk.Entry(self.sliderFrame, width=23)
        self.rtrashLabel = tk.Label(self.sliderFrame, text='Re-Identification threshold')
        self.re_identificationThreshold = tk.Entry(self.sliderFrame, width=23)

        self.videoPathLabel = tk.Label(self.sliderFrame, text='Full Path to input file')
        self.videoPathEntry = tk.Entry(self.sliderFrame, width=23)

        self.startButton = tk.Button(self.sliderFrame, bg="red", fg="#000", text="Start")
        self.startButton.bind('<1>', self.validate_fields)

        self.my_label = tk.Label(self.root)
        self.my_label.pack()
        # thread = threading.Thread(target=self.stream, args=(self.my_label, input_cap))
        # thread.daemon = 1
        # thread.start()

        #self.startButton.bind('<ButtonRelease-1>', self.stream("videcam//videcam_6.mov"))

        self.root.resizable(False, False)

        self.inputLabel.pack()
        self.inputMode.pack()
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
        
    def validate_fields(self, event):
        if self.mainMode.get() == "" or self.deviceMode.get() == "" \
            or self.re_identificationMode.get() == "" or self.aizooThreshold.get() == "" \
            or self.detectorThreshold.get() == "" or self.re_identificationThreshold.get() == "" \
            or self.videoPathEntry.get() == "":
            return
        else:
            gui_api(self)

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
            resized = self.current_image.resize((650, 420), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(resized)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            #self.panel.config(image=imgtk)  # show the image
            label1 = tk.Label(frame, image=imgtk)
            label1.image = imgtk
            label1.place(x=20, y=20)

        print("FPS: ", 1.0 / (time.time() - start_time))
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def destructor(self):
        """ Destroy the root object and release all resources """
        # print("[INFO] closing...")
        log.info("closing...")
        self.root.destroy()
        # self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

    def stream(self, label, video_name):
        video = imageio.get_reader(video_name)
        for image in video.iter_data():
            start_time = time.time()
            self.current_image = Image.fromarray(image)  # convert image for PIL
            resized = self.current_image.resize((550, 350), Image.ANTIALIAS)
            frame_image = ImageTk.PhotoImage(resized)
            label.config(image=frame_image)
            label.image = frame_image
            label.place(x=120, y=80)
            print_label = tk.Label(self.root, text=(str(round(1.0 / (time.time() - start_time)))+ '   '), bg='#C4C4C4', fg='#645F5F')
            print_label.config(font=("Courier", 24, 'bold'))
            print_label.place(x=245, y=595)
        self.root.after(30, self.stream(label, video_name))
        
    def printMaskCounter(self, masks_counter, no_masks_counter):
        print_mlabel = tk.Label(self.root, text=(str(masks_counter)+ '   '), bg='#C4C4C4', fg='#645F5F')
        print_mlabel.config(font=("Courier", 24, 'bold'))
        print_mlabel.place(x=631, y=595)
        print_nmlabel = tk.Label(self.root, text=(str(no_masks_counter)+ '   '), bg='#C4C4C4', fg='#645F5F')
        print_nmlabel.config(font=("Courier", 24, 'bold'))
        print_nmlabel.place(x=956, y=595)


if __name__ == "__main__":
    pba = Gui("videcam\\videcam2.mov", camera=False)
    pba.root.mainloop()

    """
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    # start the app
    log.info("starting...")
    pba = Gui("videcam//videcam_6.mov", camera=False)
    pba.root.mainloop()
    """
