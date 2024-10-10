import os
import cv2
import torch
import numpy as np
import argparse
from yolo.common import DetectMultiBackend
from yolo.augmentations import letterbox
from yolo.general import non_max_suppression, scale_coords

Two_Body = ['050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116',
                '117', '118', '119', '120']

def get_parser():
    parser = argparse.ArgumentParser(description = 'Parameters of Extract Person Frame') 
    parser.add_argument(
        '--sample_name_path', 
        type = str,
        default = './ntu120.txt')
    parser.add_argument(
        '--video_path', 
        type = str,
        default = './Video')
    parser.add_argument(
        '--output_path', 
        type = str,
        default = './Person_Frame')
    parser.add_argument(
        '--device', 
        type = int,
        default = 0)
    parser.add_argument(
        '--model_path', 
        type = str,
        default = './pretrained/yolov5m.pt')
    parser.add_argument(
    '--data_yaml', 
    type = str,
    default = './pretrained/coco128.yaml')
    return parser

class Detect_Person():
    def __init__(self, model_path, data_yaml, device):
        self.model_path = model_path
        self.data_yaml = data_yaml # './pretrained/coco128.yaml'
        self.device = torch.device(device)
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
    
    def init_model(self):
        model = DetectMultiBackend(self.model_path, device=self.device, dnn=False, data=self.data_yaml, fp16=False)
        return model
    
    def detect_human(self, frame_img, model, time):
        img = letterbox(frame_img, (640, 640), stride=model.stride, auto=model.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) 

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0 
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

        det = det[np.argwhere(det[:, -1] == 0), :] # 0 class: Person

        loc = np.zeros((det.shape[0], 6))
        for idx in range(det.shape[0]):
            loc[idx, :] = det[idx, :]
            loc[idx, -1] = time
        return loc
    
    def extract_person(self, img_frame, label):

        persons_locs = self.detect_human(img_frame, self.yolo_model, 0)

        if persons_locs.shape[0] == 0: # no Person
            Num = 0
            return None, None, None, Num
        
        elif (label not in Two_Body) and persons_locs.shape[0] == 1: # one Person
            x1 = int(persons_locs[0][0])
            y1 = int(persons_locs[0][1])
            x2 = int(persons_locs[0][2])
            y2 = int(persons_locs[0][3])
            img = img_frame[y1:y2, x1:x2, :]
            Num = 1
            return img, None, None, Num

        elif (persons_locs.shape[0] > 1) and (label in Two_Body) : # Two Person
            p1_x1 = int(persons_locs[0][0])
            p1_y1 = int(persons_locs[0][1])
            p1_x2 = int(persons_locs[0][2])
            p1_y2 = int(persons_locs[0][3])
            img1 = img_frame[p1_y1:p1_y2, p1_x1:p1_x2, :]

            p2_x1 = int(persons_locs[1][0])
            p2_y1 = int(persons_locs[1][1])
            p2_x2 = int(persons_locs[1][2])
            p2_y2 = int(persons_locs[1][3])
            img2 = img_frame[p2_y1:p2_y2, p2_x1:p2_x2, :]

            x1 = min(int(persons_locs[0][0]), int(persons_locs[1][0]))
            y1 = min(int(persons_locs[0][1]), int(persons_locs[1][1]))
            x2 = max(int(persons_locs[0][2]), int(persons_locs[1][2]))
            y2 = max(int(persons_locs[0][3]), int(persons_locs[1][3]))
            img3 = img_frame[y1:y2, x1:x2, :]

            Num = 2
            return img1, img2, img3, Num
        
        elif (label not in Two_Body) and persons_locs.shape[0] > 1:
            x1 = int(persons_locs[0][0])
            y1 = int(persons_locs[0][1])
            x2 = int(persons_locs[0][2])
            y2 = int(persons_locs[0][3])
            img = img_frame[y1:y2, x1:x2, :]
            Num = 1
            return img, None, None, Num
        
        elif (label in Two_Body) and persons_locs.shape[0] == 1:
            x1 = int(persons_locs[0][0])
            y1 = int(persons_locs[0][1])
            x2 = int(persons_locs[0][2])
            y2 = int(persons_locs[0][3])
            img = img_frame[y1:y2, x1:x2, :]
            Num = 2
            return img, img, img, Num

def Extract_Person_frame(samples, video_path, output_path, model_path, data_yaml, device):
    model = Detect_Person(model_path, data_yaml, device)
    for _, name in enumerate(samples):
        print("Processing " + name)
        label = name[-3:]
        setup_id = int(name[1:4])
        if int(setup_id) < 18: 
            video_file_path = video_path + '/NTU60/' + name + '_rgb.avi'
            save_path = output_path + '/NTU60/' + name + '/'
        else: 
            video_file_path = video_path + '/NTU120/' + name + '_rgb.avi'
            save_path = output_path + '/NTU120/' + name + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        cap = cv2.VideoCapture(video_file_path)
        frame_idx = 0
        ret = True
        while ret:
            ret, rgb_img = cap.read()  # read each frame
            if (not ret):
                break
            img1, img2, img3, num = model.extract_person(rgb_img, label)
            if num == 1:
                save_name = save_path + str(frame_idx) + '.jpg'
                cv2.imwrite(save_name, img1)
            elif num == 2:
                # save_name1 = save_path + str(frame_idx) + '_p1.jpg'
                # save_name2 = save_path + str(frame_idx) + '_p2.jpg'
                # save_name3 = save_path + str(frame_idx) + '_mp.jpg'
                # cv2.imwrite(save_name1, img1)
                # cv2.imwrite(save_name2, img2)
                # cv2.imwrite(save_name3, img3)
                save_name = save_path + str(frame_idx) + '.jpg'
                cv2.imwrite(save_name, img3)
            frame_idx += 1

if __name__ == "__main__":
    # nohup python Extract_person_from_video.py --sample_name_path /data-home/liujinfu/new_Yolo_Simcc/sample_txt/S01_04.txt --device 0 > /data-home/liujinfu/Extract_NTU_Person/outlog/S01_04.txt 2>&1 &
    # nohup python Extract_person_from_video.py --sample_name_path /data-home/liujinfu/new_Yolo_Simcc/sample_txt/S05_08.txt --device 0 > /data-home/liujinfu/Extract_NTU_Person/outlog/S05_08.txt 2>&1 &
    # nohup python Extract_person_from_video.py --sample_name_path /data-home/liujinfu/new_Yolo_Simcc/sample_txt/S09_12.txt --device 0 > /data-home/liujinfu/Extract_NTU_Person/outlog/S09_12.txt 2>&1 &
    # nohup python Extract_person_from_video.py --sample_name_path /data-home/liujinfu/new_Yolo_Simcc/sample_txt/S13_16.txt --device 0 > /data-home/liujinfu/Extract_NTU_Person/outlog/S13_16.txt 2>&1 &
    # nohup python Extract_person_from_video.py --sample_name_path /data-home/liujinfu/new_Yolo_Simcc/sample_txt/S17_20.txt --device 0 > /data-home/liujinfu/Extract_NTU_Person/outlog/S17_20.txt 2>&1 &
    # nohup python Extract_person_from_video.py --sample_name_path /data-home/liujinfu/new_Yolo_Simcc/sample_txt/S21_24.txt --device 0 > /data-home/liujinfu/Extract_NTU_Person/outlog/S21_24.txt 2>&1 &
    # nohup python Extract_person_from_video.py --sample_name_path /data-home/liujinfu/new_Yolo_Simcc/sample_txt/S25_28.txt --device 0 > /data-home/liujinfu/Extract_NTU_Person/outlog/S25_28.txt 2>&1 &
    # nohup python Extract_person_from_video.py --sample_name_path /data-home/liujinfu/new_Yolo_Simcc/sample_txt/S29_32.txt --device 0 > /data-home/liujinfu/Extract_NTU_Person/outlog/S29_32.txt 2>&1 &
    parser = get_parser()
    args = parser.parse_args()
    sample_name_path = args.sample_name_path
    samples = np.loadtxt(sample_name_path, dtype=str)
    video_path = args.video_path
    output_path = args.output_path
    model_path = args.model_path
    data_yaml = args.data_yaml
    device = args.device
    Extract_Person_frame(samples, video_path, output_path, model_path, data_yaml, device)
    print("All done!")