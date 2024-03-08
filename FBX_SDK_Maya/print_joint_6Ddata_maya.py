'''
@File    :   print_joint_6Ddata_maya.py
@Time    :   2024/03/07 20:05:00
@Author  :   Jinfu Liu
@Version :   1.0 
@Desc    :   print 6D data of joint in FBX file
'''

import maya.cmds as cmds

joint_names = ["root", "pelvis", "spine_00", "spine_01", "spine_02", "spine_03", "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l", "index_01_l",
        "index_02_l", "index_03_l", "middle_01_l", "middle_02_l", "middle_03_l", "pinky_01_l", "pinky_02_l", "pinky_03_l", "ring_01_l", "ring_02_l",
        "ring_03_l", "thumb_01_l", "thumb_02_l", "thumb_03_l", "Slot_hand_L_bone", "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r", "index_01_r",
        "index_02_r", "index_03_r", "middle_01_r", "middle_02_r", "middle_03_r", "pinky_01_r", "pinky_02_r", "pinky_03_r", "ring_01_r", "ring_02_r",
        "ring_03_r", "thumb_01_r", "thumb_02_r", "thumb_03_r", "Slot_hand_R_bone", "Slot_spine_bone", "neck_01", "head", "thigh_l", "calf_l","foot_l", 
        "ball_l", "thigh_r", "calf_r", "foot_r", "ball_r", "Slot_waist_L_bone", "Slot_waist_R_bone", "Slot_pelvis_bone", "ik_foot_root", "ik_foot_l",
        "ik_foot_r", "ik_hand_root", "ik_hand_gun", "ik_hand_l", "ik_hand_r"]

for joint in joint_names:
    obj = cmds.ls(joint)
    print("process ", obj)
    keyframes = cmds.keyframe(obj, query=True)
    for frame in keyframes:
        local_trans_X = cmds.getAttr(joint + ".translateX", time = frame)
        local_trans_Y = cmds.getAttr(joint + ".translateY", time = frame)
        local_trans_Z = cmds.getAttr(joint + ".translateZ", time = frame)
        local_rotate_X = cmds.getAttr(joint + ".rotateX", time = frame)
        local_rotate_Y = cmds.getAttr(joint + ".rotateY", time = frame)
        local_rotate_Z = cmds.getAttr(joint + ".rotateZ", time = frame)
        print(local_trans_X, local_trans_Y, local_trans_Z)
        print(local_rotate_X, local_rotate_Y, local_rotate_Z)
