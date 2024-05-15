import os
import cv2
import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description = 'Parameters of Extract Video frame') 
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
        default = './Video_Frame')
    return parser

def Extract_frame(sample_name, rgb_video_path, output_path):
    for _, name in enumerate(sample_name):
        print("Processing " + name)
        setup_id = int(name[1:4])
        if int(setup_id) < 18: 
            rgb_video_file = rgb_video_path + '/NTU60/' + name + '_rgb.avi'
            save_path = output_path + '/NTU60/' + name
        else: 
            rgb_video_file = rgb_video_path + '/NTU120/' + name + '_rgb.avi'
            save_path = output_path + '/NTU120/' + name
        
        cap = cv2.VideoCapture(rgb_video_file)
        frame_idx = 0
        ret = True
        while ret:
            ret, rgb_img = cap.read()  # read each frame
            if (not ret):
                break
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # cv2.imwrite(save_path + '/' + str(frame_idx) + '.jpg', rgb_img)
            cv2.imwrite(save_path + '/' + str(frame_idx) + '.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 77]) # rate = 77
            frame_idx = frame_idx + 1
    
    print("All done!")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    sample_name_path = args.sample_name_path
    sample_name = np.loadtxt(sample_name_path, dtype=str)
    rgb_video_path = args.video_path
    output_path = args.output_path

    Extract_frame(sample_name, rgb_video_path, output_path)
