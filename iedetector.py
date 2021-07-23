import cv2
import numpy as np
from openvino.inference_engine import IECore


def check_bounds(n, min_val=0, max_val=None):
    if n < min_val:
        return min_val
    elif max_val is not None and n > max_val:
        return max_val
    else:
        return n


class InferenceEngineNetwork:
    def __init__(self, configPath=None, weightsPath=None,
                 device='CPU', extension=None, classesPath=None):
        # Add code for Inference Engine initialization
        self.ie = IECore()

        # Add code for model loading
        self.net = self.ie.read_network(model=configPath, weights=weightsPath)
        self.model = configPath
        self.classes = classesPath

        # Add code for classes names loading
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

        if extension:
            self.ie.add_extension(unicode_extension_path=extension, unicode_device_name=device)

        return

    def get_top(self, prob, topN=1):
        result = prob

        # Add code for getting top predictions
        result = np.squeeze(result)
        result = np.argsort(result)[-topN:][::-1]

        return result

    def _prepare_image(self, image, h, w, interpolation=cv2.INTER_AREA):
        # Add code for image preprocessing
        image = cv2.resize(image, (w, h), interpolation=interpolation)
        image = image.transpose((2, 0, 1))

        return image

    def detect(self, image, aizoo=False):
        # Add code for image classification using Inference Engine
        input_blob = next(iter(self.net.input_info))
        # out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.input_info[input_blob].input_data.shape

        image = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: image})

        if aizoo:
            return output['loc_branch_concat'], output['cls_branch_concat']
        else:
            return output

    def expand_resolution(self, image, inter_image):
        # Add code for image classification using Inference Engine
        n, c, h, w = self.net.input_info['0'].input_data.shape
        n1, c1, h1, w1 = self.net.input_info['1'].input_data.shape

        image = self._prepare_image(image, h, w, interpolation=cv2.INTER_CUBIC)
        inter_image = self._prepare_image(inter_image, h1, w1)
        output = self.exec_net.infer(inputs={"0": image, "1": inter_image})

        return output
