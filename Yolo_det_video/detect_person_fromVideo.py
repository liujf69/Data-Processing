'''
@File    :   detect_person_fromVideo.py
@Time    :   2024/03/06 11:20:00
@Author  :   Jinfu Liu
@Version :   1.0 
@Desc    :   detect person from Video stream
'''

import os
import cv2
import torch
import numpy as np

from yolo.common import DetectMultiBackend
from yolo.torch_utils import select_device
from yolo.augmentations import letterbox
from yolo.general import non_max_suppression, scale_coords

class Yolo_Person():
    def __init__(self):
        self.device = torch.device(0)
        self.weight_pt = './pretrained/yolov5m.pt'
        self.data_yaml = './pretrained/coco128.yaml'
        self.yolo_model = self.init_model()
        
    def _xywh2cs(self, x, y, w, h, image_size):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        image_width = image_size[0]
        image_height = image_size[1]
        aspect_ratio = image_width * 1.0 / image_height
        pixel_std = 200

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
    
    # init model
    def init_model(self):
        model = DetectMultiBackend(self.weight_pt, device=self.device, dnn=False, data=self.data_yaml, fp16=False)
        return model
    
    # detect human, get location
    def detect_human(self, frame_img, model, time):

        img = letterbox(frame_img, (640, 640), stride=model.stride, auto=model.pt)[0] # letterbox scale
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0 # norm
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim 1 C H W

        pred = model(img, augment=False, visualize=False)

        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        classes = None
        agnostic_nms = False
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame_img.shape).round()
        det = det.cpu().numpy()
        det = det[np.argwhere(det[:, -1] == 0), :] # class 0: Person
        loc = np.zeros((det.shape[0], 6))
        for idx in range(det.shape[0]):
            loc[idx, :] = det[idx, :]
            loc[idx, -1] = time
        return loc
    
    def view_human(self, img_frame, persons_locs):
        for idx in range(persons_locs.shape[0]):
            # view each person
            x1 = int(persons_locs[idx][0])
            y1 = int(persons_locs[idx][1])
            x2 = int(persons_locs[idx][2])
            y2 = int(persons_locs[idx][3])
            img_frame = cv2.rectangle(img_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # color: red, thickness: 2 
        return img_frame

    def extract_person(self, img_frame):
        persons_locs = self.detect_human(img_frame, self.yolo_model, 0)
        return persons_locs

if __name__ == "__main__":
    model = Yolo_Person() # init
    video_path = './S001C001P001R001A050_rgb.avi'
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, img = cap.read()  # read each video frame
        if (not ret): # read done or read error
            break
        persons_locs = model.extract_person(img) # process each frame # may be you need to save this
        img_det = model.view_human(img, persons_locs) # persons_locs.shape: [num of person, 6]
        cv2.imwrite('./output_img/' + str(frame_idx) + '.jpg', img_det)
        frame_idx += 1