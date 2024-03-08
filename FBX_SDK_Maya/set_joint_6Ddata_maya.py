'''
@File    :   set_joint_6Ddata_maya.py
@Time    :   2024/03/07 20:10:00
@Author  :   Jinfu Liu
@Version :   1.0 
@Desc    :   set 6D data of joint in FBX file
'''

# you must install numpy by: mayapy.exe -m pip install numpy
import numpy as np
import maya.cmds as cmds

Joint_to_idx = {
    "root": 0,
    "pelvis": 1,
    "spine_00": 2,
    "spine_01": 3,
    "spine_02": 4,
    "spine_03": 5,
    "clavicle_l": 6,
    "upperarm_l": 7,
    "lowerarm_l": 8,
    "hand_l": 9,
    "index_01_l": 10,
    "index_02_l": 11,
    "index_03_l": 12,
    "middle_01_l": 13,
    "middle_02_l": 14,
    "middle_03_l": 15,
    "pinky_01_l": 16,
    "pinky_02_l": 17,
    "pinky_03_l": 18,
    "ring_01_l": 19,
    "ring_02_l": 20,
    "ring_03_l": 21,
    "thumb_01_l": 22,
    "thumb_02_l": 23,
    "thumb_03_l": 24,
    "Slot_hand_L_bone": 25,
    "clavicle_r": 26,
    "upperarm_r": 27,
    "lowerarm_r": 28,
    "hand_r": 29,
    "index_01_r": 30,
    "index_02_r": 31,
    "index_03_r": 32,
    "middle_01_r": 33,
    "middle_02_r": 34,
    "middle_03_r": 35,
    "pinky_01_r": 36,
    "pinky_02_r": 37,
    "pinky_03_r": 38,
    "ring_01_r": 39,
    "ring_02_r": 40,
    "ring_03_r": 41,
    "thumb_01_r": 42,
    "thumb_02_r": 43,
    "thumb_03_r": 44,
    "Slot_hand_R_bone": 45,
    "Slot_spine_bone": 46,
    "neck_01": 47,
    "head": 48,
    "thigh_l": 49,
    "calf_l": 50,
    "foot_l": 51,
    "ball_l": 52,
    "thigh_r": 53,
    "calf_r": 54,
    "foot_r": 55,
    "ball_r": 56,
    "Slot_waist_L_bone": 57,
    "Slot_waist_R_bone": 58,
    "Slot_pelvis_bone": 59,
    "ik_foot_root": 60,
    "ik_foot_l": 61,
    "ik_foot_r": 62,
    "ik_hand_root": 63,
    "ik_hand_gun": 64,
    "ik_hand_l": 65,
    "ik_hand_r": 66
}

Local_Trans_data = np.load("C:/Users/jinfullliu/Desktop/test_maya/Local_Trans.npy", allow_pickle = True)
local_Rotate_data = np.load("C:/Users/jinfullliu/Desktop/test_maya/local_Rotate.npy", allow_pickle = True)

for joint in Joint_to_idx:
    joint_idx = Joint_to_idx[joint]
    obj = cmds.ls(joint)
    print("process ", obj)
    for frame in range(Local_Trans_data.shape[0]):
        cmds.setKeyframe(joint + '.translateX', value = Local_Trans_data[frame, joint_idx, 0], time=frame)
        cmds.setKeyframe(joint + '.translateY', value = Local_Trans_data[frame, joint_idx, 1], time=frame)
        cmds.setKeyframe(joint + '.translateZ', value = Local_Trans_data[frame, joint_idx, 2], time=frame)
        cmds.setKeyframe(joint + '.rotateX', value = local_Rotate_data[frame, joint_idx, 0], time=frame)
        cmds.setKeyframe(joint + '.rotateY', value = local_Rotate_data[frame, joint_idx, 1], time=frame)
        cmds.setKeyframe(joint + '.rotateZ', value = local_Rotate_data[frame, joint_idx, 2], time=frame)
