import fbx
import sys
import json
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# FBX_Common function
def InitializeSdkObjects():
    """return FBX_SDK_Manager and Scene"""
    lSdkManager = fbx.FbxManager.Create()
    if not lSdkManager:
        sys.exit(0)
    # Create an IOSettings object
    ios = fbx.FbxIOSettings.Create(lSdkManager, fbx.IOSROOT)
    lSdkManager.SetIOSettings(ios)
    # Create the entity that will hold the scene.
    lScene = fbx.FbxScene.Create(lSdkManager, "")
    return (lSdkManager, lScene)

# FBX_Common function
def LoadScene(pSdkManager, pScene, pFileName):
    """Load Scene Objects"""
    lImporter = fbx.FbxImporter.Create(pSdkManager, "")    
    result = lImporter.Initialize(pFileName, -1, pSdkManager.GetIOSettings())
    if not result:
        return False
    if lImporter.IsFBX():
        pSdkManager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
        pSdkManager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
        pSdkManager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
        pSdkManager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_SHAPE, True)
        pSdkManager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GOBO, True)
        pSdkManager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_ANIMATION, True)
        pSdkManager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GLOBAL_SETTINGS, True)
    result = lImporter.Import(pScene)
    lImporter.Destroy()
    return result

class Joint:
    def __init__(self) -> None:
        self.name = ""
        self.global_translation = None
        self.global_rotation = None
        self.global_scale = None
        self.children = None
        self.local_translation = None
        self.local_rotation = None
        self.local_scale = None

    def __call__(self):
        return{
            "name": self.name,
            "global_translation": self.global_translation,
            "global_rotation": self.global_rotation,
            "global_scale": self.global_scale,
            "children": self.children,
            "local_translation": self.local_translation,
            "local_rotation": self.local_rotation,
            "local_scale": self.local_scale
        }

class Parser:
    def __init__(self) -> None:
        print("init FBX Parser")
        
    def load_fbx(self, fbx_file: str):
        self.sdk_manager, self.scene = InitializeSdkObjects()
        status_flag = LoadScene(self.sdk_manager, self.scene, fbx_file)
        if status_flag is None:
            warnings.warn("Err: Load Scene Failed.")
            return None

        geoCount = self.scene.GetGeometryCount() # the count of geometry
        unit = self.scene.GetGlobalSettings().GetSystemUnit().GetScaleFactorAsString() # unit, e.g. m or cm
        anim_stack = self.scene.GetMember(fbx.FbxAnimStack.ClassId) # get sequence data
        if anim_stack is None:
            warnings.warn("Err: sequence data is None")
            return None
        anim_layer = anim_stack.GetMember(fbx.FbxAnimLayer.ClassId)
        if anim_layer is None:
            warnings.warn("Err: anim_layer is None")
            return None
        
        # time object
        time_span = anim_stack.GetLocalTimeSpan()
        start_time = time_span.GetStart()
        end_time = time_span.GetStop()

        frame_rate = fbx.FbxTime.GetFrameRate(anim_stack.GetScene().GetGlobalSettings().GetTimeMode()) # frame rate, FPS
        total_frames = end_time.GetFrameCount() - start_time.GetFrameCount() # total frames
        root_node = self.scene.GetRootNode() # the root node

        joint_on_Allframes = [] # joint datas of all frames
        for frame in tqdm(range(total_frames)): # for each frame
            time = fbx.FbxTime()  # time
            time.SetFrame(frame, fbx.FbxTime.EMode.eFrames30)  # current frame
            joints_list = self.get_joints(root_node, time) # get all joints of one frame
            joint_on_Allframes.append(joints_list)

        # global datas of one fbx file
        status_info = {
            "geo_count": geoCount,
            "scene_unit": unit,
            "total_frames": total_frames,
            "frame_rate": frame_rate,
            "joint_datas": joint_on_Allframes
        }
        return status_info

    def get_joints(self, node, time):
        """
        input:
            node    the joint class
            time    the time class
        return:
            list
        """
        if node is None:
            warnings.warn("Err: the input node is None!")
            return []
        joints = []
        node_attr = node.GetNodeAttribute() # get attribute
        if node_attr and node_attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
            joint = Joint()
            joint.name = node.GetName() # the name of joint

            # global
            # generally the 6D pose is the translation and rotation 
            g_translation = node.EvaluateGlobalTransform(time).GetT() # translation
            g_rotation = node.EvaluateGlobalTransform(time).GetR() # rotation
            g_scale = node.EvaluateGlobalTransform(time).GetS() # scale
            joint.global_translation = list(g_translation)
            joint.global_rotation = list(g_rotation)
            joint.global_scale = list(g_scale)
            # local
            l_translation = node.EvaluateLocalTransform(time).GetT() # translation
            l_rotation = node.EvaluateLocalTransform(time).GetR() # rotation
            l_scale = node.EvaluateLocalTransform(time).GetS() # scale
            joint.local_translation = list(l_translation)
            joint.local_rotation = list(l_rotation)
            joint.local_scale = list(l_scale)

            for i in range(node.GetChildCount()): # for each child node
                child = node.GetChild(i)
                child_attr = child.GetNodeAttribute() # get attribute
                if child_attr and child_attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton: # the skeleton node, other like mesh and so on
                    joint.children = child.GetName() if child.GetName() else "NULL"
            joints.append(joint)

        for i in range(node.GetChildCount()): # for each child node
            child = node.GetChild(i)
            sub_joints = self.get_joints(child, time) # recursion
            joints.extend(sub_joints)
        return joints
    
    # draw global trans of one frame
    def draw_gtrans(self, frame_joints):
        len_joints = len(frame_joints)

        # record all joint names
        name_joint_list = []
        for joint_idx in range(len_joints):
            name_joint_list.append(frame_joints[joint_idx].name)

        name_map = [] # e.g. root:0, pelvis:1, ...
        bone_link = []
        fig = plt.figure()
        plot3D = plt.subplot(projection = '3d')
        for joint_idx in range(len_joints): # for each joint
            joint_name = frame_joints[joint_idx].name
            name_map.append({joint_name:joint_idx})
            if frame_joints[joint_idx].children != None: # have children node
                bone_link.append([joint_idx, name_joint_list.index(frame_joints[joint_idx].children)])
            
            # draw scatter
            position = frame_joints[joint_idx].global_translation
            plot3D.scatter(position[0], position[2], position[1], c = 'red', s = 2.0) # in unity, y is up

        # draw bone link
        for bone_idx in range(len(bone_link)):
            start = bone_link[bone_idx][0]
            end = bone_link[bone_idx][1]
            pos_start = frame_joints[start].global_translation
            pos_end = frame_joints[end].global_translation
            plot3D.plot([pos_start[0], pos_end[0]], # x 
                        [pos_start[2], pos_end[2]], # z # in unity, y is up
                        [pos_start[1], pos_end[1]], # y
                        c = 'green', lw = 2)
        # plt.savefig('./vis_test.png')
        return fig

def main():
    my_parser = Parser()
    fbx_file = './Anim_H_9101_M_Stand_01_001_To_Stand_01_002.fbx'
    status_info = my_parser.load_fbx(fbx_file)

    # draw one frame data
    frame_idx = 0
    fig = my_parser.draw_gtrans(status_info['joint_datas'][frame_idx])
    fig.savefig('./test.png')

if __name__ == "__main__":
    main()