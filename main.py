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
    'C:\\Program Files (x86)\\Intel\openvino_2021.4.582\\deployment_tools\\open_model_zoo\\demos\\common\python')
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
    output = cv2.VideoWriter('output_camera.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1280, 720))
    while True:
        image = cap.read()

        y_bboxes_output, y_cls_output = ie_classifier.classify(image)
        image = post_processing(image, y_bboxes_output, y_cls_output)

        cv2.imshow("result", image)
        output.write(image)

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

    #single_image_processing(ie_classifier, args.input)
    image_capture_processing(ie_classifier, args.input)
    return


if __name__ == '__main__':
    sys.exit(main())
