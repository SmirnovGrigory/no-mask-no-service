import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


def check_min(n):
    if n < 0:
        return 0
    else:
        return n


def check_max(n, maximum):
    if n > maximum:
        return maximum
    else:
        return n


class InferenceEngineClassifier:
    def __init__(self, configPath=None, weightsPath=None,
                 device='CPU', extension=None, classesPath=None):
        # Add code for Inference Engine initialization
        self.ie = IECore()

        # Add code for model loading
        self.net = self.ie.read_network(model=configPath)

        # Add code for classes names loading
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

        return

    def get_top(self, prob, topN=1):
        result = prob

        # Add code for getting top predictions
        result = np.squeeze(result)
        result = np.argsort(result)[-topN:][::-1]

        return result

    def _prepare_image(self, image, h, w):
        # Add code for image preprocessing
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))

        return image

    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.inputs[input_blob].shape

        image = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: image})

        out = output['detection_out'][0]

        width = 568
        height = 320
        for detection in out[0]:
            x_min = detection[3] * width
            y_min = detection[4] * height
            x_max = detection[5] * width
            y_max = detection[6] * height
            x_delta = x_max - x_min
            y_delta = y_max - y_min
            x_min -= (0.3 * x_delta)
            y_min -= (0.3 * y_delta)
            x_max += (0.3 * x_delta)
            y_max += (0.3 * y_delta)
            detection[3] = check_min(x_min)
            detection[4] = check_min(y_min)
            detection[5] = check_max(x_max, width)
            detection[6] = check_max(y_max, height)

        return out[0]

    def classify(self, image):
        probabilities = None

        # Add code for image classification using Inference Engine
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.inputs[input_blob].shape

        image = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: image})

        return (output['loc_branch_concat'], output['cls_branch_concat'])
