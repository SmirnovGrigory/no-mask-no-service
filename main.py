import os
import cv2
import sys
import argparse
import logging as log
from openvino.inference_engine import IENetwork, IECore
from Postprocessing import post_processing
from ieclassifier import InferenceEngineClassifier


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


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    #log.info("Start IE classification sample")

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start IE classification sample")
    ie_classifier = InferenceEngineClassifier(configPath=args.model,
                                              weightsPath=args.weights,
                                              device=args.device,
                                              extension=args.cpu_extension,
                                              classesPath=args.classes)
    image = cv2.imread(args.input)
    y_bboxes_output, y_cls_output = ie_classifier.classify(image)
    #log.info(y_bboxes_output)
    image = post_processing(image, y_bboxes_output, y_cls_output)

    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # predictions = ie_classifier.get_top(prob, 5)
    # log.info("Predictions: " + str(predictions))

    return


if __name__ == '__main__':
    sys.exit(main())
