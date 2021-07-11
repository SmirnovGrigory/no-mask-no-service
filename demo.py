import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore

sys.path.append(
    'C:\\Program Files (x86)\\Intel\openvino_2021.4.582\\deployment_tools\\open_model_zoo\\demos\\common\python')
import models
from pipelines import AsyncPipeline
from images_capture import open_images_capture


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-w', '--weights', help='Path to an .bin file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    parser.add_argument('-l', '--cpu_extension', help='MKLDNN \
        (CPU)-targeted custom layers.Absolute path to a shared library \
        with the kernels implementation', type=str, default=None)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    return parser


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified


def draw_detections(frame, detections, labels, threshold):
    # TODO
    size = frame.shape[:2]

    with open(labels, 'r') as f:
        classes_list = f.read().split('\n')
    label_map = dict(enumerate(classes_list))

    for detection in detections:
        # If score more than threshold, draw rectangle on the frame
        score = detection.score
        if score > threshold:
            cv2.rectangle(frame, (int(detection.xmin), int(detection.ymin)), (int(detection.xmax), int(detection.ymax)),
                          (0, 255, 0), 1)
            cv2.putText(frame, label_map[detection.id] + " " + "{:.4f}".format(score),
                        (int(detection.xmin), int(detection.ymin)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)

    return frame

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start OpenVINO object detection")

    # Initialize data input
    cap = open_images_capture(args.input, True)

    # Initialize OpenVINO
    ie = IECore()

    # Initialize Plugin configs
    plugin_configs = get_plugin_configs('CPU', 0, 0)


    

    # Destroy all windows
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    sys.exit(main())
