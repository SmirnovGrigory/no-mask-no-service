import cv2
import sys
import argparse
import logging as log
import time
import pathlib
from openvino.inference_engine import IECore
from postprocessing import post_processing, calc_features_similarity
from iedetector import InferenceEngineNetwork, check_bounds
from PIL import Image, ImageTk

sys.path.append(
    'C:\\Program Files (x86)\\Intel\\openvino_2021.4.582\\deployment_tools\\open_model_zoo\\demos\\common\\python')

import models
from pipelines import AsyncPipeline
from images_capture import open_images_capture


def sign_case(case):
    if case:
        return 1
    else:
        return -1


def draw_detections(frame, detections, labels, threshold, *, reid_threshold=0.9, draw_result=False,
                    mask_net=None,
                    reid_list=None, reid_net=None):
    label_map = None
    if labels:
        with open(labels, 'r') as f:
            classes_list = f.read().split('\n')
        label_map = dict(enumerate(classes_list))

    height, width = frame.shape[:2]

    for detection in detections:
        score = detection.score

        extend_coef = 0.3
        x_delta = detection.x_max - detection.x_min
        y_delta = detection.y_max - detection.y_min
        detection.x_min -= (extend_coef * x_delta)
        detection.y_min -= (extend_coef * y_delta)
        detection.x_max += (extend_coef * x_delta)
        detection.y_max += (extend_coef * y_delta)
        detection[3] = check_bounds(detection.x_min)
        detection[4] = check_bounds(detection.y_min)
        detection[5] = check_bounds(detection.x_max, max=width)
        detection[6] = check_bounds(detection.y_max, max=height)

        # If score more than threshold, draw rectangle on the frame
        if score > threshold if threshold > 0 else 0.1:
            face_image = frame[int(detection[4]):int(detection[6]),
                               int(detection[3]):int(detection[5])]

            if reid_net is not None and reid_list is not None:
                out = reid_net.detect(face_image)
                for old_note in reid_list:
                    if calc_features_similarity(out, old_note) > reid_threshold if reid_threshold > 0 else 0.1:
                        break
                else:
                    reid_list.append(out)
            elif reid_list:
                reid_list.append(detection)
            else:
                reid_list = [detection]

            pred = None
            if mask_net is not None:
                y_bboxes_output, y_cls_output = mask_net.detect(face_image, aizoo=True)
                pred = post_processing(face_image, y_bboxes_output, y_cls_output, just_pred=True, draw_result=True)

            if draw_result:
                cv2.rectangle(frame, (int(detection.xmin), int(detection.ymin)), (int(detection.xmax),
                                                                                  int(detection.ymax)),
                              (0, 255, 0), 1)
                if labels:
                    cv2.putText(frame, f"Object {label_map[detection.id]} with {detection.score:.2f}",
                                (int(detection.xmin), int(detection.ymin - 10)), cv2.FONT_HERSHEY_COMPLEX,
                                0.45, (0, 0, 255), 1)
                else:
                    cv2.putText(frame, f"{pred['pred'] if mask_net is not None else 'Object with'} {score:.2f}",
                                (int(detection[3]), int(detection[4] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.45,
                                (0, 0, 255), 1)

    return reid_list


def draw_detections_with_postprocessing(frame, detections, labels, threshold, *, draw_result=False, reid_list=None,
                                        reid_net=None, reid_threshold=0.6, aizoo_threshold=0.5,
                                        mask_net=None):
    label_map = None
    if labels:
        with open(labels, 'r') as f:
            classes_list = f.read().split('\n')
        label_map = dict(enumerate(classes_list))
    height, width = frame.shape[:2]
    for detection in detections['detection_out'][0][0]:

        # frame_data = dict()
        # feature_names = ['score', 'x_min', 'y_min', 'x_max', 'y_max']
        # for i in range(2, len(detection)):
        # frame_data[feature_names[i-2]] = detection[i] if i < 3 else
        # detection[i] * (width if i % 2 == 1 else height)
        score = detection[2]
        x_min = detection[3] * width
        y_min = detection[4] * height
        x_max = detection[5] * width
        y_max = detection[6] * height
        x_delta = x_max - x_min
        y_delta = y_max - y_min
        extend_coef = 0.3
        # for i, feature in enumerate(['x_min', 'y_min', 'x_max', 'y_max']):
        #     frame_data[feature] -= sign_case(i < 2) * (extend_coef * (x_delta if i % 2 == 0 else y_delta))
        #     detection[i + 3] = check_bounds(frame_data[feature],
        #                                     max=((width if i % 2 == 0 else height) if i > 1 else None))
        x_min -= (extend_coef * x_delta)
        y_min -= (extend_coef * y_delta)
        x_max += (extend_coef * x_delta)
        y_max += (extend_coef * y_delta)
        detection[3] = check_bounds(x_min)
        detection[4] = check_bounds(y_min)
        detection[5] = check_bounds(x_max, max=width)
        detection[6] = check_bounds(y_max, max=height)

        # If score more than threshold, draw rectangle on the frame
        if score > threshold if threshold > 0 else 0.1:
            face_image = frame[int(detection[4]):int(detection[6]),
                               int(detection[3]):int(detection[5])]

            cv2.imshow("face", face_image)

            if mask_net is None and reid_net is not None and reid_list is not None:
                out = reid_net.detect(face_image)
                for old_note in reid_list:
                    if calc_features_similarity(out, old_note) > reid_threshold if reid_threshold > 0 else 0.1:
                        break
                else:
                    reid_list.append(out)
            elif mask_net is None and reid_list is not None:
                reid_list.append(detection)
            elif mask_net is None:
                reid_list = [detection]

            pred = None
            if mask_net is not None:
                y_bboxes_output, y_cls_output = mask_net.detect(face_image, aizoo=True)
                pred, reid_list = post_processing(face_image, y_bboxes_output, y_cls_output, just_pred=True,
                                                  draw_result=False,
                                                  reid_list=reid_list, reid_net=reid_net, conf_thresh=aizoo_threshold)

            if draw_result:
                cv2.rectangle(frame, (int(detection[3]), int(detection[4])),
                              (int(detection[5]), int(detection[6])), (0, 255, 0), 1)
                if labels:
                    cv2.putText(frame, f"Object {label_map[detection.id]} with {detection[2]:.2f}",
                                (int(detection[3]), int(detection[4] - 10)), cv2.FONT_HERSHEY_COMPLEX,
                                0.45, (0, 0, 255), 1)
                else:
                    cv2.putText(frame, f"{pred['pred'] if mask_net is not None else 'Face with'} {score:.2f}",
                                (int(detection[3]), int(detection[4] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.45,
                                (0, 0, 255), 1)

    return reid_list


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Paths to .xml \
        files with a trained models.', nargs='+', type=str)
    parser.add_argument('-w', '--weights', help='Paths to .bin files \
        with a trained weights.', nargs='+', type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image or video file, to use your webcam just type smth with "cam"',
                        required=True, type=str)
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


def single_image_processing(detector, image_path, gui=None,
                            in_model_api=False,
                            write_me=False,
                            resolution='None', resolution_net=None,
                            mask_net=None,
                            reid_list=None, reid_net=None):
    image = cv2.imread(image_path)

    if in_model_api:
        frame_id = 0
        detector.submit_data(image, frame_id, {'frame': image, 'start_time': 0})

        # Wait for processing finished
        detector.await_any()

        # Get detection result
        results, meta = detector.get_result(frame_id)
        if gui is not None:
            reid_list = draw_detections(image, results, None, float(gui.detectorThreshold.get()),
                                        reid_threshold=float(gui.re_identificationThreshold.get()))
        else:
            reid_list = draw_detections(image, results, None, 0.5,
                                        reid_threshold=0.5)
    else:
        if "face_mask_detection" in detector.model or "AIZOO" in detector.model:
            y_bboxes_output, y_cls_output = detector.detect(image, aizoo=True)
            if gui is not None:
                image, reid_list = post_processing(
                    image, y_bboxes_output, y_cls_output, reid_list=reid_list,
                    reid_net=reid_net,
                    reid_threshold=float(gui.re_identificationThreshold.get()),
                    conf_thresh=float(gui.aizooThreshold.get()))
            else:
                image, reid_list = post_processing(
                    image, y_bboxes_output, y_cls_output, reid_list=reid_list,
                    reid_net=reid_net,
                    reid_threshold=0.5,
                    conf_thresh=0.5)
        else:
            output = detector.detect(image)
            if gui is not None:
                reid_list = draw_detections_with_postprocessing(
                    image, output, None,
                    float(gui.detectorThreshold.get()), reid_list=reid_list,
                    reid_net=reid_net, draw_result=True, mask_net=mask_net,
                    reid_threshold=float(gui.re_identificationThreshold.get()))
            else:
                reid_list = draw_detections_with_postprocessing(
                    image, output, None, 0.5,
                    reid_list=reid_list,
                    reid_net=reid_net, draw_result=True, mask_net=mask_net,
                    reid_threshold=0.5)

    if resolution == 'all' and resolution_net is not None:
        image = next(iter(resolution_net.expand_resolution(image, image).values()))[0]
        image = image.transpose((1, 2, 0))

    if gui is None:
        cv2.imshow("result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gui.current_image = Image.fromarray(image)  # convert image for PIL
        resized = gui.current_image.resize((550, 350))

        frame_image = ImageTk.PhotoImage(resized, Image.ANTIALIAS)
        gui.my_label.config(image=frame_image)
        gui.my_label.image = frame_image
        gui.my_label.place(x=120, y=80)

    if mask_net is not None or 'AIZOO' in detector.model:
        with_mask, without_mask = 0, 0
        if reid_list is not None:
            for reid_dict in reid_list:
                if reid_dict['pred'] == 'NoMask':
                    without_mask += 1
                else:
                    with_mask += 1
        log.info(f"There are {with_mask} people with mask and {without_mask} people without mask on the photo")
        if gui is not None:
            gui.printMaskCounter(with_mask, without_mask)
    else:
        if reid_list is not None:
            log.info(f"{len(reid_list)} peoples on the photo")
            gui.printMaskCounter(0, len(reid_list))
        else:
            log.info(f"0 peoples on the photo")
            gui.printMaskCounter(0, 0)

    if write_me:
        cv2.imwrite('results\\result_img' + str(round(time.time())) + '.jpg', image)


# def image_capture_processing(detector, input_cap, in_model_api=False):
#     cap = open_images_capture(input_cap, True)
#     output = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1280, 720))
#     while True:
#         image = cap.read()
#
#         if in_model_api:
#             frame_id = 0
#             detector.submit_data(image, frame_id, {'frame': image, 'start_time': 0})
#
#             # Wait for processing finished
#             detector.await_any()
#
#             # Get detection result
#             results, meta = detector.get_result(frame_id)
#             draw_detections(image, results, None, 0.5)
#         else:
#             if "face_mask_detection" in detector.xml or "AIZOO" in detector.xml:
#                 y_bboxes_output, y_cls_output = detector.detect(image, aizoo=True)
#                 image = post_processing(image, y_bboxes_output, y_cls_output)
#             else:
#                 output = detector.detect(image)
#                 draw_detections_with_postprocessing(image, output, None, 0.8)
#
#         cv2.imshow("result", image)
#         # output.write(image)
#
#         # Wait 1 ms and check pressed button to break the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

def video_processing(detector, input_cap=None, gui=None,
                     in_model_api=False,
                     write_me=False,
                     resolution='None', resolution_net=None,
                     mask_net=None,
                     reid_list=None, reid_net=None):
    if input_cap is not None:
        cap = open_images_capture(input_cap, True)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

    output = None
    if write_me:
        output = cv2.VideoWriter('output_camera.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1280, 720))

    # gui.my_label.place(x=120, y=80)

    # gui.stream(gui.my_label, 'videcam\\videcam_5.mov')

    while True:
        # if gui is not None:
        #     image = next(iter(cap.iter_data()))
        start_time = time.time()
        if input_cap is not None:
            image = cap.read()
        else:
            ret, image = cap.read()
            image = cv2.flip(image, 1)

        if in_model_api:
            frame_id = 0
            detector.submit_data(image, frame_id, {'frame': image, 'start_time': 0})

            # Wait for processing finished
            detector.await_any()

            # Get detection result
            results, meta = detector.get_result(frame_id)
            reid_list = draw_detections(image, results, None, float(gui.detectorThreshold.get()),
                                        reid_threshold=float(gui.re_identificationThreshold.get()))
        else:
            if "face_mask_detection" in detector.model or "AIZOO" in detector.model:
                y_bboxes_output, y_cls_output = detector.detect(image, aizoo=True)
                image, reid_list = post_processing(
                    image, y_bboxes_output, y_cls_output, reid_list=reid_list,
                    reid_net=reid_net, reid_threshold=float(gui.re_identificationThreshold.get()),
                    conf_thresh=float(gui.aizooThreshold.get()))
            else:
                output = detector.detect(image)
                reid_list = draw_detections_with_postprocessing(
                    image, output, None, float(gui.detectorThreshold.get()),
                    reid_list=reid_list,
                    reid_net=reid_net, draw_result=True, mask_net=mask_net,
                    reid_threshold=float(gui.re_identificationThreshold.get()),
                    aizoo_threshold=float(gui.aizooThreshold.get()))

        if resolution == 'hd' and resolution_net is not None:
            image = next(iter(resolution_net.expand_resolution(image, image).values()))[0]
            image = image.transpose((1, 2, 0))

        # cv2.imshow("Result frame", image)
        if gui is None:
            cv2.imshow("result", image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gui.current_image = Image.fromarray(image)  # convert image for PIL
            resized = gui.current_image.resize((550, 350))

            frame_image = ImageTk.PhotoImage(resized, Image.ANTIALIAS)
            gui.my_label.config(image=frame_image)
            gui.my_label.image = frame_image
            gui.my_label.place(x=120, y=80)
            gui.root.update()

        gui.printFps(start_time)

        if write_me:
            output.write(image)

        if mask_net is not None or 'AIZOO' in detector.model:
            with_mask, without_mask = 0, 0
            if reid_list is not None:
                for reid_dict in reid_list:
                    if reid_dict['pred'] == 'NoMask':
                        without_mask += 1
                    else:
                        with_mask += 1
            log.info(f"There are {with_mask} people with mask and {without_mask} people without mask on the photo")
            gui.printMaskCounter(with_mask, without_mask)
        else:
            if reid_list is not None:
                log.info(f"{len(reid_list)} peoples on the photo")
                gui.printMaskCounter(0, len(reid_list))
            else:
                log.info(f"0 peoples on the photo")
                gui.printMaskCounter(0, 0)

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

    reidentification_list = []
    in_model_api = False
    model_api_names = ["faceboxes", "retinaface", "ssd", "yolo", "centernet"]
    models_list = []
    detector, reidentificator, resolutioner, mask_detector = None, None, None, None

    for model_name, model_weights in zip(args.model, args.weights):
        for model_api_name in model_api_names:
            if model_api_name in model_name:
                ie = IECore()
                # HETERO:CPU,GPU or CPU or GPU
                plugin_configs = get_plugin_configs(args.device, 0, 0)
                detector_model = models.FaceBoxes(ie, pathlib.Path(model_name), input_transform)
                models_list.append(AsyncPipeline(ie, detector_model, plugin_configs,
                                                 device=args.device,
                                                 max_num_requests=1))

                in_model_api = True
                break
        else:
            models_list.append(InferenceEngineNetwork(configPath=model_name,
                                                      weightsPath=model_weights,
                                                      device=args.device,
                                                      extension=args.cpu_extension,
                                                      classesPath=args.classes))
        last_created_model = models_list[-1:][0]
        if 'mask_detection' in model_name:
            mask_detector = last_created_model
        elif 'detection' in model_name:
            detector = last_created_model
        elif 'reidentification' in model_name:
            reidentificator = last_created_model
        elif 'resolution' in model_name:
            resolutioner = last_created_model

    if 'cam' in args.input and '.' not in args.input:
        args.input = None

    if args.input is not None and \
            args.input.endswith(('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'gif')):
        single_image_processing(mask_detector, args.input, in_model_api=in_model_api,
                                reid_list=reidentification_list, resolution='None',
                                resolution_net=resolutioner, mask_net=None,
                                reid_net=reidentificator)
    elif args.input is not None and \
            args.input.endswith(('avi', 'wmv', 'mov', 'mkv', '3gp', '264', 'mp4')) or \
            args.input is None:
        video_processing(detector, input_cap=args.input, in_model_api=in_model_api,
                         reid_list=reidentification_list, resolution='None',
                         resolution_net=resolutioner, mask_net=None,
                         reid_net=reidentificator)
    else:
        raise Exception('unknown or invalid input format')

    return


def gui_api(gui):
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)

    reidentification_list = []
    in_model_api = False
    deviceMode = gui.deviceMode.get()
    precision = "INT8"
    if deviceMode == "HETERO:CPU,GPU":
        precision = "FP16"

    mask_detector = InferenceEngineNetwork(
        configPath=f'models\\model-AIZOO\\{precision}\\face_mask_detection.xml',
        weightsPath=f'models\\model-AIZOO\\{precision}\\face_mask_detection.bin',
        device=deviceMode,
        extension=None,
        classesPath=None)
    detector = InferenceEngineNetwork(
        configPath=f"models\\face-detection-0200\\{precision}\\face-detection-0200.xml",
        weightsPath=f"models\\face-detection-0200\\{precision}\\face-detection-0200.bin",
        device=deviceMode,
        extension=None,
        classesPath=None)
    reidentificator = None
    if gui.re_identificationMode.get() == "Re-Identification On":
        reidentificator = InferenceEngineNetwork(
            configPath=f"models\\face-reidentification-retail-0095\\{precision}\\face-reidentification-retail-0095.xml",
            weightsPath=f"models\\face-reidentification-retail-0095\\{precision}\\"
                        f"face-reidentification-retail-0095.bin",
            device=deviceMode,
            extension=None,
            classesPath=None)

    if gui.inputMode.get() == "Image":
        if gui.mainMode.get() == "AIZOO":
            single_image_processing(mask_detector, gui.videoPathEntry.get(), gui=gui, in_model_api=in_model_api,
                                    reid_list=reidentification_list, resolution='None',
                                    resolution_net=None, mask_net=None,
                                    reid_net=reidentificator)
        elif gui.mainMode.get() == "Face Detector":
            single_image_processing(detector, gui.videoPathEntry.get(), gui=gui, in_model_api=in_model_api,
                                    reid_list=reidentification_list, resolution='None',
                                    resolution_net=None, mask_net=None,
                                    reid_net=reidentificator)
        elif gui.mainMode.get() == "Face Detector + AIZOO":
            single_image_processing(detector, gui.videoPathEntry.get(), gui=gui, in_model_api=in_model_api,
                                    reid_list=reidentification_list, resolution='None',
                                    resolution_net=None, mask_net=mask_detector,
                                    reid_net=reidentificator)

    elif gui.inputMode.get() == "Video":
        if gui.mainMode.get() == "AIZOO":
            video_processing(mask_detector, input_cap=gui.videoPathEntry.get(), gui=gui, in_model_api=in_model_api,
                             reid_list=reidentification_list, resolution='None',
                             resolution_net=None, mask_net=None,
                             reid_net=reidentificator)
        elif gui.mainMode.get() == "Face Detector":
            video_processing(detector, input_cap=gui.videoPathEntry.get(), gui=gui, in_model_api=in_model_api,
                             reid_list=reidentification_list, resolution='None',
                             resolution_net=None, mask_net=None,
                             reid_net=reidentificator)
        elif gui.mainMode.get() == "Face Detector + AIZOO":
            video_processing(detector, input_cap=gui.videoPathEntry.get(), gui=gui, in_model_api=in_model_api,
                             reid_list=reidentification_list, resolution='None',
                             resolution_net=None, mask_net=mask_detector,
                             reid_net=reidentificator)
    elif gui.inputMode.get() == "Web camera":
        if gui.mainMode.get() == "AIZOO":
            video_processing(mask_detector, gui=gui, in_model_api=in_model_api,
                             reid_list=reidentification_list, resolution='None',
                             resolution_net=None, mask_net=None,
                             reid_net=reidentificator)
        elif gui.mainMode.get() == "Face Detector":
            video_processing(detector, gui=gui, in_model_api=in_model_api,
                             reid_list=reidentification_list, resolution='None',
                             resolution_net=None, mask_net=None,
                             reid_net=reidentificator)
        elif gui.mainMode.get() == "Face Detector + AIZOO":
            video_processing(detector, gui=gui, in_model_api=in_model_api,
                             reid_list=reidentification_list, resolution='None',
                             resolution_net=None, mask_net=mask_detector,
                             reid_net=reidentificator)
    else:
        raise Exception('unknown input format')


if __name__ == '__main__':
    sys.exit(main())
