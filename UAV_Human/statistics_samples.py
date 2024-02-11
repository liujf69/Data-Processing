import os
import numpy as np

def get_samples(skeleton_path: str) -> None:
    samples = sorted(os.listdir(Skeleton_path))
    assert len(samples) == 23031
    samples_list = []
    for sample in samples:
        samples_list.append(sample)
    np.savetxt('./all_samples.txt', samples_list, fmt = "%s")
    
def get_max_frame(samples_txt_path: str) -> int:
    root_Skeleton_path = './Skeleton'
    max_frame = 0
    samples_txt = np.loadtxt(samples_txt_path, dtype = str)
    for idx, sample in enumerate(samples_txt):
        ske_path = root_Skeleton_path + '/' + sample
        with open(ske_path, 'r') as f:
            cur_frame = int(f.readline()) # the frame num
            if cur_frame > max_frame: max_frame = cur_frame
    return max_frame  
  
if __name__ == "__main__":
    Skeleton_path = './Skeleton'
    get_samples(Skeleton_path)
    samples_txt_path = './all_samples.txt'
    max_frame = get_max_frame(samples_txt_path)
    print("max frame is: ", max_frame) # 305
    
    