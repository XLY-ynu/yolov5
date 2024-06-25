import numpy as np
import torch
from openvino.inference_engine import IECore
import cv2
from utils.general import non_max_suppression, scale_coords, xyxy2xywh


def combine_outputs(outputs):
    y0, y1, y2 = outputs

    y0 = y0.reshape((1, -1, y0.shape[-1]))
    y1 = y1.reshape((1, -1, y1.shape[-1]))
    y2 = y2.reshape((1, -1, y2.shape[-1]))

    combined_output = np.concatenate((y0, y1, y2), axis=1)

    return combined_output


def run_inference(weights, input_image):
    ie = IECore()
    net = ie.read_network(model=weights)
    exec_net = ie.load_network(network=net, device_name='CPU')

    input_blob = next(iter(net.input_info))
    output_blobs = list(net.outputs.keys())

    im = preprocess_image(input_image)

    request = exec_net.start_async(request_id=0, inputs={input_blob: im})
    request.wait()

    outputs = [request.output_blobs[blob].buffer for blob in output_blobs]

    combined_output = combine_outputs(outputs)

    return combined_output


def preprocess_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.resize(im, (640, 640))
    im = im.transpose((2, 0, 1))  # HWC to CHW
    im = np.expand_dims(im, axis=0)
    im = im / 255.0  # Normalize to [0, 1]
    return im


def postprocess(combined_output, conf_thres=0.25, iou_thres=0.45, max_det=1000):
    combined_output = torch.tensor(combined_output)
    pred = non_max_suppression(combined_output, conf_thres, iou_thres, max_det=max_det)
    return pred


def visualize_detections(image_path, detections, names):
    im0 = cv2.imread(image_path)
    for det in detections:
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=generate_color(int(cls)))
    cv2.imshow("Detections", im0)
    cv2.waitKey(0)


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image im
    tl = line_thickness or round(0.002 * max(im.shape[0:2]))  # line thickness
    color = [int(c) for c in color]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def generate_color(idx):
    # Generate a unique color for each class index
    np.random.seed(idx)
    color = [int(c) for c in np.random.randint(0, 255, 3)]
    return color


# Example usage
weights = 'path/to/your/model.xml'
input_image = 'path/to/your/image.jpg'
combined_output = run_inference(weights, input_image)
detections = postprocess(combined_output)

# Load class names (e.g., from coco.names)
names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
         'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
         'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
         'teddy bear', 'hair drier', 'toothbrush']

visualize_detections(input_image, detections, names)
