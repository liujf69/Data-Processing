'''
@File    :   Extract_local_TR.py
@Time    :   2024/03/08 15:22:00
@Author  :   Jinfu Liu
@Version :   1.0 
@Desc    :   Extract skeleton data from FBX file for Maya visualization
'''

import os
import fbx
from tqdm import tqdm
import numpy as np

Mujin_JOINT_TO_IDX = {
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

class FBX_Parser():
    def __init__(self, filename = None):
        '''
        初始化导入的FBX文件
        '''
        self.manager = fbx.FbxManager.Create()
        self.scene = fbx.FbxScene.Create(self.manager, "")
        if filename:
            self.import_from_file(filename)

    def import_from_file(self, filename):
        '''
        打印导入FBX文件的部分信息
        '''
        importer = fbx.FbxImporter.Create(self.manager, "")
        importStatus = importer.Initialize(filename)
        if not importStatus:
            raise Exception(f"Importer failed to initialized.\nError returned: {importer.GetStatus().GetErrorString()}")

        major, minor, revision = importer.GetFileVersion()
        print(f"version numbers of {os.path.basename(filename)}: {major}.{minor}.{revision}")

        importer.Import(self.scene)
        importer.Destroy()

        timeModeStrings = ["DefaultMode", "Frames120", "Frames100", "Frames60", "Frames50", "Frames48",
            "Frames30", "Frames30Drop", "NTSCDropFrame", "NTSCFullFrame", "PAL", "Frames24", "Frames1000",
            "FilmFullFrame", "Custom", "Frames96", "Frames72", "Frames59dot94"]

        globalSettings = self.scene.GetGlobalSettings()
        self.timeMode = globalSettings.GetTimeMode()
        self.frameRate = fbx.FbxTime().GetFrameRate(globalSettings.GetTimeMode())
        self.timeModeString = timeModeStrings[self.timeMode.value]
        timeSpan = globalSettings.GetTimelineDefaultTimeSpan()
        self.start, self.stop = timeSpan.GetStart(), timeSpan.GetStop()
        print(f"Global time settings: {self.timeModeString}, {self.frameRate} fps")
        print(f"Timeline default timespan: [{self.start.GetTimeString()}: {self.stop.GetTimeString()}]")

    def evaluate_animation_transforms(self):
        '''
        function: 提取Maya对应的平移坐标和欧拉旋转角坐标
        '''
        animLayer = self.get_first_available_anim_layer() # 获取第一个可用的动画层
        start, stop = self.get_animlayer_frame_span(animLayer) # 获取动画层的开始帧和结束帧

        self.hierarchy = self.get_skeleton_hierarchy() # 获取人体骨骼的树形层次结构
        non_endsite_hierarchy = self.hierarchy

        frameCount = stop - start + 1 # 总帧数
        jointCount = len(non_endsite_hierarchy)

        # 提取对应Maya软件中的平移X、平移Y和平移Z
        self.localTrans = np.zeros((frameCount, jointCount, 3))
        for f in tqdm(range(start, stop + 1)): # for each frame
            time = fbx.FbxTime() # set frame
            time.SetFrame(f, self.timeMode)
            for i in range(jointCount): # for each joint
                node = non_endsite_hierarchy[i]["node"]
                ltrans = node.EvaluateLocalTransform(time).GetT() # get frame data
                self.localTrans[f, i, :] = list(ltrans)[:3] # save 

        # 提取对应Maya软件中的旋转X、旋转Y和旋转Z，即Euler坐标
        self.EulerRotate = np.zeros((frameCount, jointCount, 3))
        for i in range(jointCount): # for each joint
            node = non_endsite_hierarchy[i]["node"]
            scene = node.GetScene()
            anim_stack = scene.GetCurrentAnimationStack()
            anim_layer = anim_stack.GetMember(0)
            lcl_rotation = node.LclRotation
            curve_node = lcl_rotation.GetCurveNode(anim_layer)

            for f in tqdm(range(start, stop + 1)): # for each frame
                value_x = curve_node.GetCurve(0).KeyGetValue(f) # value X, frame f 
                value_y = curve_node.GetCurve(1).KeyGetValue(f) # value Y, frame f 
                value_z = curve_node.GetCurve(2).KeyGetValue(f) # value Y, frame f 
                self.EulerRotate[f, i, :] = [value_x, value_y, value_z] # save

        return self.localTrans, self.EulerRotate
    
    def get_first_available_anim_layer(self):
        '''
        function: 获取第一个可用的动画层
        '''
        animStackCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
        animLayerCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId)
        animCurveNodeCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
        animStackCount = self.scene.GetSrcObjectCount(animStackCriteria)
        for i in range(animStackCount):
            animStack = self.scene.GetSrcObject(animStackCriteria, i)
            animLayerCount = animStack.GetMemberCount(animLayerCriteria)
            for j in range(animLayerCount):
                animLayer = animStack.GetMember(animLayerCriteria, j)
                animCurveNodeCount = animLayer.GetMemberCount(animCurveNodeCriteria)
                if animCurveNodeCount > 0:
                    print(f"Selected AnimLayer: [{animStack.GetName()}, {animLayer.GetName()} ({animCurveNodeCount} Curves)]")
                    return animLayer
        return None
    
    def get_animlayer_frame_span(self, animLayer):
        '''
        function: 获取动画层的开始帧和结束帧
        '''
        animCurveNodeCriteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimCurveNode.ClassId)
        animCurveNodeCount = animLayer.GetMemberCount(animCurveNodeCriteria)
        animLayerStart, animLayerStop = self.start, self.stop
        for i in range(animCurveNodeCount):
            animCurveNode = animLayer.GetMember(animCurveNodeCriteria, i)
            interval = fbx.FbxTimeSpan()
            animCurveNode.GetAnimationInterval(interval)
            start, stop = interval.GetStart(), interval.GetStop()
            animLayerStart = min(animLayerStart, start)
            animLayerStop = max(animLayerStop, stop)
        return (animLayerStart.GetFrameCount(self.timeMode), animLayerStop.GetFrameCount(self.timeMode))
    
    def get_skeleton_hierarchy(self):
        '''
        获取骨骼的树形层次结构
        '''
        skeletonRoot = self.get_skeleton_root() # 获取骨骼的root节点
        hierarchy = []

        def is_zero_rotation(r):
            return np.allclose(r, [0, 0, 0])

        def depth_first_iterate_outliner(node, i, parent, hierarchy):
            localTransform = node.EvaluateLocalTransform()
            T, R = localTransform.GetT(), localTransform.GetR()

            item = {
                "name": node.GetName(),
                "node": node,
                "offset": (T[0], T[1], T[2]),
                "rotation": (R[0], R[1], R[2]),
                "skeleton": isinstance(node.GetNodeAttribute(), fbx.FbxSkeleton),
                "children": [],
                "children_indices": [],
                "parent": parent,
                "parent_index": -1,
                "birth_order": i,
                "is_end_site": False,
                "sibling": 1 if not parent else parent.GetChildCount(),
            }
            hierarchy.append(item)

            childCount = node.GetChildCount()
            for i in range(childCount):
                child = node.GetChild(i)
                item["children"].append(child)
                item["children_indices"].append(-1)
                depth_first_iterate_outliner(child, i, node, hierarchy)

            item["is_end_site"] = (
                len(item["children_indices"]) == 0 and item["birth_order"] == 0 and is_zero_rotation(item["rotation"])
            )

        depth_first_iterate_outliner(skeletonRoot, 0, None, hierarchy)
        nodes = [item["node"] for item in hierarchy]
        for item in hierarchy:
            if item["parent"]:
                index = nodes.index(item["parent"])
                item["parent_index"] = index

            for nth in range(len(item["children"])):
                index = nodes.index(item["children"][nth])
                item["children_indices"][nth] = index

        if not hasattr(self, "hierarchy_checked"):
            nonSkeletonNodes = [item["node"].GetName() for item in hierarchy if not item["skeleton"]]
            if len(nonSkeletonNodes) > 0:
                print(
                    f"Caution! Non-Skeleton nodes exist in the hierarchy ({len(nonSkeletonNodes)} / {len(hierarchy)}, {len(hierarchy) - len(nonSkeletonNodes)})."
                )
                print([item["node"].GetName() for item in hierarchy if not item["skeleton"]])
            self.hierarchy_checked = True

        return hierarchy
    
    def get_skeleton_root(self):
        '''
        返回第一个符合fbx.FbxSkeleton类型的节点
        '''
        nodeQueue = [self.scene.GetRootNode()]
        while len(nodeQueue) > 0:
            node = nodeQueue.pop(0)
            if isinstance(node.GetNodeAttribute(), fbx.FbxSkeleton): 
                return node
            childCount = node.GetChildCount()
            for i in range(childCount):
                child = node.GetChild(i)
                nodeQueue.append(child)
        return None

if __name__ == "__main__":
    fbx_path = './Anim_H_9102_F_Stand_01_Loop_BendSalute02.fbx'
    parser = FBX_Parser(fbx_path)
    localTrans, EulerRotate = parser.evaluate_animation_transforms()

    np.save("./Local_Trans.npy", localTrans)
    np.save("./local_Rotate.npy.npy", EulerRotate)