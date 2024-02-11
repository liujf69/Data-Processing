import numpy as np
import matplotlib.pyplot as plt

def extract_pose(ske_txt_path: str) -> np.ndarray:
    with open(ske_txt_path, 'r') as f: 
        num_frame = int(f.readline()) # the frame num
        joint_data = [] # T M V C
        for t in range(num_frame): # for each frame
            num_body = int(f.readline()) # the body num
            one_frame_data = np.zeros((num_body, 17, 2)) # M 17 2 
            for m in range(num_body): # for each body
                f.readline() # skip this line, e.g. 000 0 0 0 0 0 0 0 0 0
                num_joints = int(f.readline()) # the num joins, equal to 17
                assert num_joints == 17
                for v in range(num_joints): # for each joint
                    xy = np.array(f.readline().split()[:2], dtype = np.float64)
                    one_frame_data[m, v] = xy
            joint_data.append(one_frame_data)
        joint_data = np.array(joint_data)  
    return joint_data # T M 17 2 
    
def vis_pose(joint_data: np.ndarray) -> None:
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
    }
    '''
    # skeleton = [
    #     [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
    #     [7, 9], [8, 10], [9, 11], [1, 2], [1, 3], [2, 4], [3, 5], [1, 7], [1, 6]
    # ]
    x_min = np.min(joint_data[:, :, :, 0])
    x_max = np.max(joint_data[:, :, :, 0])
    y_min = np.min(joint_data[:, :, :, 1])
    y_max = np.max(joint_data[:, :, :, 1]) 
    bone_connect1 = [0, 1, 3] # [1, 2] [2, 4]
    bone_connect2 = [0, 2, 4] # [1, 3] [3, 5]
    bone_connect3 = [0, 5, 6, 12, 11, 13, 15] # [1, 6] [6, 7] [7, 13] [13, 12] [12, 14] [14, 16]
    bone_connect4 = [0, 6, 8, 10] # [1, 7] [7, 9] [9, 11]
    bone_connect5 = [11, 5, 7, 9] # [12, 6] [6, 8] [8, 10]
    bone_connect6 = [12, 14, 16] # [13, 15] [15, 17]
    num_frame = joint_data.shape[0] # # T M 17 2 
    for t in range(num_frame):
        plt.figure() # new figure
        # Draw scatter points
        plt.scatter(joint_data[t, :, :, 0], joint_data[t, :, :, 1], c = 'red', s = 40.0) 
        
        # Draw body 1
        plt.plot(joint_data[t, 0, bone_connect1, 0], joint_data[t, 0, bone_connect1, 1], c = 'green', lw = 2.0)
        plt.plot(joint_data[t, 0, bone_connect2, 0], joint_data[t, 0, bone_connect2, 1], c = 'green', lw = 2.0)
        plt.plot(joint_data[t, 0, bone_connect3, 0], joint_data[t, 0, bone_connect3, 1], c = 'green', lw = 2.0)
        plt.plot(joint_data[t, 0, bone_connect4, 0], joint_data[t, 0, bone_connect4, 1], c = 'green', lw = 2.0)
        plt.plot(joint_data[t, 0, bone_connect5, 0], joint_data[t, 0, bone_connect5, 1], c = 'green', lw = 2.0)
        plt.plot(joint_data[t, 0, bone_connect6, 0], joint_data[t, 0, bone_connect6, 1], c = 'green', lw = 2.0)
        
        if joint_data.shape[1] == 2:
            # Draw body 2
            plt.plot(joint_data[t, 1, bone_connect1, 0], joint_data[t, 1, bone_connect1, 1], c = 'green', lw = 2.0)
            plt.plot(joint_data[t, 1, bone_connect2, 0], joint_data[t, 1, bone_connect2, 1], c = 'green', lw = 2.0)
            plt.plot(joint_data[t, 1, bone_connect3, 0], joint_data[t, 1, bone_connect3, 1], c = 'green', lw = 2.0)
            plt.plot(joint_data[t, 1, bone_connect4, 0], joint_data[t, 1, bone_connect4, 1], c = 'green', lw = 2.0)
            plt.plot(joint_data[t, 1, bone_connect5, 0], joint_data[t, 1, bone_connect5, 1], c = 'green', lw = 2.0)
            plt.plot(joint_data[t, 1, bone_connect6, 0], joint_data[t, 1, bone_connect6, 1], c = 'green', lw = 2.0)
    
        # Set the starting point to the upper left corner
        plt.xlim(x_min - 1, x_max + 1) 
        plt.ylim(y_max + 1, y_min - 1)
        if joint_data.shape[1] == 1:
            plt.savefig('./vis_imgs/one_body/' + str(t) + '.jpg')
        else:
            assert joint_data.shape[1] == 2
            plt.savefig('./vis_imgs/two_body/' + str(t) + '.jpg')
        plt.close()
    
if __name__ == "__main__":
    # one body
    ske_txt_path1 = "./Skeleton/P000S00G10B10H10UC022000LC021000A000R0_08241716.txt"
    joint_data1 = extract_pose(ske_txt_path1)
    vis_pose(joint_data1)
    # two body
    ske_txt_path2 = "./Skeleton/P000S02G11B00H00UC022072LC021061A083R0_08301122.txt"
    joint_data2 = extract_pose(ske_txt_path2)
    vis_pose(joint_data2)
    
    print("All done!")