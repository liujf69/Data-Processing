import os
import csv
import cv2

def find_video_name(root_video_path, label, youtube_id):
    video_path = os.path.join(root_video_path, label)
    video_names = os.listdir(video_path)
    for idx, video_name in enumerate(video_names):
        if(video_name[:11] == youtube_id):
            return video_name

def split_frame(raw_csv: str, root_video_path: str, root_output_path: str, debug: bool = True):
    csv_reader = csv.reader(open(raw_csv))
    for idx, row in enumerate(csv_reader): 
        if (idx == 0):
            continue # ['label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc'] 
        label, youtube_id, time_start, time_end, split, is_cc = row
        video_name = find_video_name(root_video_path, label, youtube_id)
        print("Process ", idx, " ", video_name)
        
        video_path = os.path.join(root_video_path, label, video_name)
        save_path = os.path.join(root_output_path, label, video_name.split(".")[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        ret = True
        while ret:
            ret, rgb_img = cap.read()  # read each frame
            if (not ret):
                break
            cv2.imwrite(save_path + '/' + str(frame_idx) + '.jpg', rgb_img)
            frame_idx = frame_idx + 1
        
        if debug: # just process one video
            break

if __name__ == "__main__":
    raw_csv = "./label/train_256.csv"
    root_video_path = "./raw-part/compress/train_256"
    root_output_path = "./test"
    split_frame(raw_csv = raw_csv, root_video_path = root_video_path, root_output_path = root_output_path, debug = True)

    print("All Done!")