import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


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
        probabilities = None

        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.inputs[input_blob].shape

        image = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: image})

        out = output['detection_out'][0]

        width = 568
        height = 320
        for detection in out[0]:
            detection[3] *= width
            detection[4] *= height
            detection[5] *= width
            detection[6] *= height

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