import pose_hrnet

from yolo.augmentations import letterbox
from yolo.common import DetectMultiBackend
from yolo.transforms import get_affine_transform
from yolo.general import non_max_suppression, scale_coords

import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from config import cfg
from config import update_config

import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        default = './pretrained/w48_256x192_adam_lr1e-3_split2_sigma4.yaml',
                        help = 'experiment configure file name',
                        type = str)

    parser.add_argument('opts',
                        help = "Modify config options using the command-line",
                        default = None,
                        nargs = argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help = 'model directory',
                        type = str,
                        default = '')
    parser.add_argument('--logDir',
                        help = 'log directory',
                        type = str,
                        default = '')
    parser.add_argument('--dataDir',
                        help = 'data directory',
                        type = str,
                        default = '')
    parser.add_argument('--prevModelDir',
                        help = 'prev Model directory',
                        type = str,
                        default = '')

    args = parser.parse_args()
    return args

class HP_estimation():
    def __init__(self):
        self.model_detect, self.model_pose = self.init_model()

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
        args = parse_args()
        update_config(cfg, args)

        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
        model_pose = eval(cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
        model_pose.load_state_dict(torch.load('./pretrained/pose_hrnet_w48_256x192_split2_sigma4.pth', map_location='cuda:0'))
        model_pose = torch.nn.DataParallel(model_pose, device_ids=(0,)).cuda(0)
        device = torch.device('cuda:0')
        model_detect = DetectMultiBackend('./pretrained/yolov5m.pt', device=device, dnn=False, data='./pretrained/coco128.yaml', fp16=False)

        return model_detect, model_pose # YoloV5, SimCC

    # use yolov5 to detect human
    def detect_human(self, frame_img, model, time):
        img = letterbox(frame_img, (640, 640), stride=model.stride, auto=model.pt)[0] # letterbox缩放
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快

        device = torch.device('cuda:0')
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0 # 归一化
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

    # 姿态估计，针对每个人估计姿态
    def estimate_pose(self, data_numpy, model_pose, loc):
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        x = loc[0]  # x1
        y = loc[1]  # y1
        w = loc[2] - loc[0]  # x2-x1
        h = loc[3] - loc[1]  # y2-y1
        image_size = (data_numpy.shape[0], data_numpy.shape[1])
        c, s = self._xywh2cs(x, y, w, h, image_size)
        r = 0
        image_size = (192, 256)

        trans = get_affine_transform(c, s, r, image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        # show = copy.deepcopy(input)

        # Data loading code
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        input = transform(input)
        input = input[np.newaxis, :, :, :]
        input = input.to(device = 0)
        pred_x, pred_y = model_pose(input) # pred pose

        idx_x = pred_x.argmax(2)
        idx_y = pred_y.argmax(2)

        idx_x = idx_x.cpu().float().numpy().squeeze(0)
        idx_y = idx_y.cpu().float().numpy().squeeze(0)

        idx_x /= cfg.MODEL.SIMDR_SPLIT_RATIO
        idx_y /= cfg.MODEL.SIMDR_SPLIT_RATIO

        '''
        "keypoints": {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle"
        },
        '''
        # skeleton = [
        #     [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
        #     [7, 9], [8, 10], [9, 11], [1, 2], [1, 3], [2, 4], [3, 5], [1, 7], [1, 6]
        # ]
        # for idx in skeleton:
        #     st_x = int(idx_x[idx[0] - 1])
        #     st_y = int(idx_y[idx[0] - 1])
        #     ed_x = int(idx_x[idx[1] - 1])
        #     ed_y = int(idx_y[idx[1] - 1])
        #     cv2.line(show, (st_x, st_y), (ed_x, ed_y), (0, 255, 0), 2)

        # return pred_x, pred_y, idx_x, idx_y, show, feature1, feature2, feature3, feature4
        return pred_x, pred_y, idx_x, idx_y

    # 结合目标检测和姿态估计的总体实现
    def extract_feature_from_one_person(self, model, model_pose, img_frame):
        persons_locs = self.detect_human(img_frame, model, 0) # detect Person
        if persons_locs.shape[0] == 0: # no Person
            return None, None

        preds = []
        skeletons = np.zeros((persons_locs.shape[0], 1, 17, 2), dtype = np.float32) # M 1 17 2
        heats_x = np.zeros((persons_locs.shape[0], 1, 17, 384))
        heats_y = np.zeros((persons_locs.shape[0], 1, 17, 512))

        for idx, loc1 in enumerate(persons_locs): # select one person for pose estimation 
            h = 256
            w = 192
            skeleton = np.zeros((1, 17, 2))
            heat_x, heat_y, idx_x1, idx_y1 = self.estimate_pose(img_frame, model_pose, loc1) 
            # heatx.shape: [1, 17, 384]
            # heaty.shape: [1, 17, 512]   
            # feature1.shape: [1, 48(C), 64, 48]
            # feature.shape: [1, 161, 64, 48] 
            # feature = torch.cat((feature1, feature2, feature3, feature4), dim = 1)

            skeleton[0, :, 0] = idx_x1
            skeleton[0, :, 1] = idx_y1
            skeleton = skeleton.astype(np.float32) #[1, 17, 2]
            skeletons[idx] = skeleton
            heats_x[idx] = heat_x.cpu().float().detach().numpy()
            heats_y[idx] = heat_y.cpu().float().detach().numpy()

        data_dict = {"skeleton": skeletons, "location": persons_locs, "heat_x": heats_x, "heat_y": heats_y} 
        return data_dict

    def HPose_estimation(self, img_frame):
        data_dict = self.extract_feature_from_one_person(self.model_detect, self.model_pose, img_frame)
        return data_dict
    
    def get_origin(self, loc, idx_x, idx_y):
        x = loc[0]  # x1
        y = loc[1]  # y1
        w = loc[2] - loc[0]  # x2-x1
        h = loc[3] - loc[1]  # y2-y1
        image_size = (1080, 1920)
        c, s = self._xywh2cs(x, y, w, h, image_size)
        r = 0
        image_size = (192, 256)
        trans = get_affine_transform(c, s, r, image_size)
        inv_trans = self.inv_align(trans)

        origin_idx_x = np.zeros((17)) # 17
        origin_idx_y = np.zeros((17)) # 17
        for idx in range(17):
            origin_idx_x[idx] = idx_x[idx]*inv_trans[0][0] + idx_y[idx]*inv_trans[0][1] + inv_trans[0][2]
            origin_idx_y[idx] = idx_x[idx]*inv_trans[1][0] + idx_y[idx]*inv_trans[1][1] + inv_trans[1][2]
        
        return origin_idx_x, origin_idx_y
    
    def inv_align(self, M):
        # M的逆变换
        k  = M[0, 0]
        b1 = M[0, 2]
        b2 = M[1, 2]
        return np.array([[1/k, 0, -b1/k], [0, 1/k, -b2/k]])

    def view_pose(self, img, origin_idx_x, origin_idx_y):
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [1, 2], [1, 3], [2, 4], [3, 5], [1, 7], [1, 6]
        ]
        for idx in skeleton:
            st_x = int(origin_idx_x[idx[0] - 1])
            st_y = int(origin_idx_y[idx[0] - 1])
            ed_x = int(origin_idx_x[idx[1] - 1])
            ed_y = int(origin_idx_y[idx[1] - 1])
            cv2.line(img, (st_x, st_y), (ed_x, ed_y), (0, 255, 0), 2)

        # for i in range(17):
        #     cv2.circle(img, (int(origin_idx_x[i]), int(origin_idx_y[i])), 5, (0,255,0), -1)
        return img

if __name__ == "__main__":
    
    model = HP_estimation()
    samples_txt = './test.txt'
    videos_path = './data/video/'
    frames_path = './data/video2frame/'
    save_vis_path = './data/Visualization/'
    save_path = "./output/"
    samples_name = np.loadtxt(samples_txt, dtype=str)

    for _, name in enumerate(samples_name): # for each sample
        print("Processing " + name)
        rgb_video_path = videos_path + name + '_rgb.avi'
        cap = cv2.VideoCapture(rgb_video_path)

        save_frames_path = frames_path + name
        if not os.path.exists(save_frames_path):
            os.makedirs(save_frames_path)

        tmp_data = []
        frame_idx = 0
        while True:
            ret, img = cap.read()  # read each video frame
            if (not ret): # read done or read error
                break

            # save frame
            cv2.imwrite(save_frames_path + '/' + str(frame_idx) + '.jpg', img)#, [cv2.IMWRITE_JPEG_QUALITY, 77]) # rate = 77

            # get pose
            data_dict = model.HPose_estimation(img) # data_dict:[T, dict]
            tmp_data.append(data_dict)
            
            # vis pose (get the origin_x and origin_y)
            loc = data_dict['location'][0]
            idx_x = data_dict['skeleton'][0][0][:, 0]
            idx_y = data_dict['skeleton'][0][0][:, 1]
            origin_x, origin_y = model.get_origin(loc, idx_x, idx_y)
            vis_img = model.view_pose(img, origin_x, origin_y)
            save_vis_img_path = save_vis_path + name
            if not os.path.exists(save_vis_img_path):
                os.makedirs(save_vis_img_path)
            cv2.imwrite(save_vis_img_path + '/' + str(frame_idx) + '.jpg', vis_img)
            
            frame_idx += 1
            
        np.save(save_path + name + ".npy", tmp_data) # save data
 
    print("All done!")
