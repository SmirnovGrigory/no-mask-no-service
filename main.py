import os
import cv2
import sys
import argparse
import logging as log
import time
from openvino.inference_engine import IENetwork, IECore
from postprocessing import post_processing
from ieclassifier import InferenceEngineClassifier

sys.path.append(
    'C:\\Program Files (x86)\\Intel\\openvino_2021.4.582\\deployment_tools\\open_model_zoo\\demos\\common\\python')
from images_capture import open_images_capture

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (255, 0, 0))

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


def single_image_processing(ie_classifier, input_img):
    image = cv2.imread(input_img)
    y_bboxes_output, y_cls_output = ie_classifier.classify(image)

    image = post_processing(image, y_bboxes_output, y_cls_output)

    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('results\\result_img' + str(round(time.time())) + '.jpg', image)


def image_capture_processing(ie_classifier, input_cap):
    cap = open_images_capture(input_cap, True)
    output = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1280, 720))
    while True:
        image = cap.read()

        y_bboxes_output, y_cls_output = ie_classifier.classify(image)
        image = post_processing(image, y_bboxes_output, y_cls_output)

        cv2.imshow("result", image)
        output.write(image)

        # Wait 1 ms and check pressed button to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def camera_capture_processing(ie_classifier, write_me=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    output = None
    if write_me:
        output = cv2.VideoWriter('output_camera.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1280, 720))

    while True:
        ret, image = cap.read()
        cv2.imshow("Current frame", image)
        y_bboxes_output, y_cls_output = ie_classifier.classify(image)
        image = post_processing(image, y_bboxes_output, y_cls_output)

        cv2.imshow("Result frame", image)
        if write_me:
            output.write(image)

        # Wait 1 ms and check pressed button to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def face_detection(ie, input_cap):
    cap = open_images_capture(input_cap, True)
    while True:
        image = cap.read()

        detections = ie.detect(image)

        for detection in detections:
            if detection[2] > 0.5:
                cv2.rectangle(image, (int(detection[3]), int(detection[4])),
                              (int(detection[5]), int(detection[6])),
                              (0, 255, 0), 1)
        cv2.imshow("result", image)

        # Wait 1 ms and check pressed button to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def face_and_mask_detection_from_image_capture(ie_detector, ie_classifier, input_cap):
    cap = open_images_capture(input_cap, True)
    while True:
        image = cap.read()

        detections = ie_detector.detect(image)

        for detection in detections:
            if detection[2] > 0.5:
                x_min = int(detection[3])
                y_min = int(detection[4])
                x_max = int(detection[5])
                y_max = int(detection[6])
                crop_image = image[y_min:y_max, x_min:x_max]

                y_bboxes_output, y_cls_output = ie_classifier.classify(image)
                pp_result = post_processing(image, y_bboxes_output, y_cls_output, draw_result=False)
                if pp_result == -1:
                    continue
                else:
                    class_id, confidence = pp_result
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[class_id], 1)
                    cv2.putText(image, "%s: %.2f" % (id2class[class_id], confidence), (x_min + 2, y_min - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id])

        cv2.imshow("result", image)

        # Wait 1 ms and check pressed button to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)

    args = build_argparser().parse_args()

    ie_classifier = InferenceEngineClassifier(configPath=args.model,
                                              weightsPath=args.weights,
                                              device=args.device,
                                              extension=args.cpu_extension,
                                              classesPath=args.classes)

    ie_detector = InferenceEngineClassifier(configPath='intel\\face-detection-adas-0001\\FP16\\face-detection-adas'
                                                       '-0001.xml',
                                            weightsPath='intel\\face-detection-adas-0001\\FP16\\face-detection-adas'
                                                        '-0001.bin',
                                            device='CPU',
                                            extension=None,
                                            classesPath=None)

    # if args.input.endswith(('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'gif')):
    #     single_image_processing(ie_classifier, args.input)
    # elif args.input.endswith(('avi', 'wmv', 'mov', 'mkv', '3gp', '264', 'mp4')):
    #     image_capture_processing(ie_classifier, args.input)
    # elif 'cam' in args.input:
    #     camera_capture_processing(ie_classifier)
    # else:
    #     raise Exception('unknown input format')
    # return

    face_and_mask_detection_from_image_capture(ie_detector, ie_classifier, args.input)


if __name__ == '__main__':
    sys.exit(main())
