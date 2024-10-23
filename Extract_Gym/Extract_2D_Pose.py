import pickle

if __name__ == "__main__":
    data_path = "./gym_hrnet.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.loads(f.read())
    
    annotations = data['annotations']
    for idx, sample in enumerate(annotations):
        frame_dir = sample['frame_dir']
        label = sample['label']
        skeleton = sample['keypoint'] # M T V 2
        frames = sample['total_frames']
        assert frames == skeleton.shape[1]
        print("debug pause")
    
    print("All Done!")