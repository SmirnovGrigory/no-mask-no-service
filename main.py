import os
import cv2
import sys
import argparse
import logging as log
import time
import pathlib
from openvino.inference_engine import IENetwork, IECore
from postprocessing import post_processing
from ieclassifier import InferenceEngineClassifier

sys.path.append(
    'C:\\Program Files (x86)\\Intel\\openvino_2021.4.582\\deployment_tools\\open_model_zoo\\demos\\common\\python')
import models
from pipelines import AsyncPipeline
from images_capture import open_images_capture


def draw_detections(frame, detections, labels, threshold):

    if labels:
        with open(labels, 'r') as f:
            classes_list = f.read().split('\n')
        label_map = dict(enumerate(classes_list))

    size = frame.shape[:2]

    for detection in detections:
        score = detection.score

        # If score more than threshold, draw rectangle on the frame
        if score > threshold:
            cv2.rectangle(frame, (int(detection.xmin), int(detection.ymin)), (int(detection.xmax), int(detection.ymax)), (0, 255, 0), 1)
            if labels:
                cv2.putText(frame, f"Object {label_map[detection.id]} with {detection.score:.2f}", (int(detection.xmin), int(detection.ymin - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
            else:
                cv2.putText(frame, f"Object with {detection.score:.2f}",
                            (int(detection.xmin), int(detection.ymin - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.45,
                            (0, 0, 255), 1)

    return frame

def draw_detections_1(frame, detections, labels, threshold):
    if labels:
        with open(labels, 'r') as f:
            classes_list = f.read().split('\n')
        label_map = dict(enumerate(classes_list))
    size = frame.shape[:2]
    for detection in detections['detection_out'][0][0]:

        score = detection[2]
        detection[3] *= size[1]
        detection[4] *= size[0]
        detection[5] *= size[1]
        detection[6] *= size[0]

        # If score more than threshold, draw rectangle on the frame
        if score > threshold:
            cv2.rectangle(frame, (int(detection[3]), int(detection[4])), (int(detection[5]), int(detection[6])), (0, 255, 0), 1)
            if labels:
                pass
                #cv2.putText(frame, f"Object {label_map[detection.id]} with {detection.score:.2f}", (int(detection.xmin), int(detection.ymin - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
            else:
                cv2.putText(frame, f"Object with {score:.2f}",
                            (int(detection[3]), int(detection[4] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.45,
                            (0, 0, 255), 1)

    return frame


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


def single_image_processing(detector, image_path, in_model_api=False):
    image = cv2.imread(image_path)

    if in_model_api:
        frame_id = 0
        detector.submit_data(image, frame_id, {'frame': image, 'start_time': 0})

        # Wait for processing finished
        detector.await_any()

        # Get detection result
        results, meta = detector.get_result(frame_id)
        draw_detections(image, results, None, 0.5)
    else:
        if "face_mask_detection" in detector.xml or "AIZOO" in detector.xml:
            y_bboxes_output, y_cls_output = detector.classify(image, aizoo=True)
            image = post_processing(image, y_bboxes_output, y_cls_output)
        else:
            output = detector.classify(image)
            draw_detections_1(image, output, None, 0.8)

    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('results\\result_img' + str(round(time.time())) + '.jpg', image)


def image_capture_processing(detector, input_cap, in_model_api=False):
    cap = open_images_capture(input_cap, True)
    output = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1280, 720))
    while True:
        image = cap.read()

        if in_model_api:
            frame_id = 0
            detector.submit_data(image, frame_id, {'frame': image, 'start_time': 0})

            # Wait for processing finished
            detector.await_any()

            # Get detection result
            results, meta = detector.get_result(frame_id)
            draw_detections(image, results, None, 0.5)
        else:
            if "face_mask_detection" in detector.xml or "AIZOO" in detector.xml:
                y_bboxes_output, y_cls_output = detector.classify(image, aizoo=True)
                image = post_processing(image, y_bboxes_output, y_cls_output)
            else:
                output = detector.classify(image)
                draw_detections_1(image, output, None, 0.8)

        cv2.imshow("result", image)
        #output.write(image)

        # Wait 1 ms and check pressed button to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def camera_capture_processing(detector, write_me=False, in_model_api=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    output = None
    if write_me:
        output = cv2.VideoWriter('output_camera.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1280, 720))

    while True:
        ret, image = cap.read()

        if in_model_api:
            frame_id = 0
            detector.submit_data(image, frame_id, {'frame': image, 'start_time': 0})

            # Wait for processing finished
            detector.await_any()

            # Get detection result
            results, meta = detector.get_result(frame_id)
            draw_detections(image, results, None, 0.5)
        else:
            if "face_mask_detection" in detector.xml or "AIZOO" in detector.xml:
                y_bboxes_output, y_cls_output = detector.classify(image, aizoo=True)
                image = post_processing(image, y_bboxes_output, y_cls_output)
            else:
                output = detector.classify(image)
                draw_detections_1(image, output, None, 0.8)

        #cv2.imshow("Current frame", image)
        cv2.imshow("Result frame", image)
        if write_me:
            output.write(image)

        # Wait 1 ms and check pressed button to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

def input_transform(arg):
    return arg

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)

    args = build_argparser().parse_args()

    in_model_api = False
    detector_pipeline = None
    ie_classifier = None
    model_api_names = ["faceboxes", "retinaface", "ssd", "yolo", "centernet"]
    for model_api_name in model_api_names:
        if model_api_name in args.model:
            ie = IECore()
            #HETERO: CPU, GPU or CPU or GPU
            plugin_configs = get_plugin_configs('CPU', 0, 0)
            detector = models.FaceBoxes(ie, pathlib.Path(args.model), input_transform)
            detector_pipeline = AsyncPipeline(ie, detector, plugin_configs,
                                              device='CPU',
                                              max_num_requests=1)
            in_model_api = True
            break
    else:
        ie_classifier = InferenceEngineClassifier(configPath=args.model,
                                                  weightsPath=args.weights,
                                                  device=args.device,
                                                  extension=args.cpu_extension,
                                                  classesPath=args.classes)

    if args.input.endswith(('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'gif')):
        single_image_processing(detector_pipeline if in_model_api else ie_classifier, args.input, in_model_api)
    elif args.input.endswith(('avi', 'wmv', 'mov', 'mkv', '3gp', '264', 'mp4')):
        image_capture_processing(detector_pipeline if in_model_api else ie_classifier, args.input, in_model_api)
    elif 'cam' in args.input:
        camera_capture_processing(detector_pipeline if in_model_api else ie_classifier, in_model_api=in_model_api)
    else:
        raise Exception('unknown input format')
    return


if __name__ == '__main__':
    sys.exit(main())
