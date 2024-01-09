import json
import mano # pip install git+'https://github.com/otaheri/MANO'
import torch
from mano.utils import Mesh

# 15个目标手关节点
target_joins = ['index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3',
                'pinky1', 'pinky2', 'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3']

def random_pose(batch_size, n_comps):
    betas = torch.rand(batch_size, 10)*.1
    pose = torch.rand(batch_size, n_comps)*.1
    global_orient = torch.rand(batch_size, 3)
    transl = torch.rand(batch_size, 3)
    return betas, pose, global_orient, transl

def load_pose(data_path, batch_size, n_comps):
    # load_data
    pose = torch.zeros(batch_size, n_comps)*.1
    with open(data_path) as json_file:
        data = json.load(json_file)
    data_pose = data['response']['result'][0]['rotations'] # 第一帧

    for idx in range(len(target_joins)):
        pose[:, idx*3:(idx+1)*3] = torch.tensor(data_pose['right_' + target_joins[0]])[1:]
    
    betas = torch.rand(batch_size, 10)*.1
    global_orient = torch.rand(batch_size, 3)
    transl = torch.rand(batch_size, 3)
    return betas, pose, global_orient, transl

if __name__ == "__main__":
    model_path = './models/mano/'
    n_comps = 45 # 15*3
    batch_size = 1
    random = True
    load_data = True
    # 导入模型
    rh_model = mano.load(model_path = model_path,
                        is_rhand = True,
                        num_pca_comps = n_comps,
                        batch_size = batch_size,
                        flat_hand_mean = False)
    if random:
        betas, pose, global_orient, transl = random_pose()
    if load_data:
        data_path = './test_json.json'
        betas, pose, global_orient, transl = load_pose(data_path, batch_size, n_comps)

    output = rh_model(betas = betas, global_orient = global_orient, hand_pose = pose,
                    transl = transl, return_verts = True, return_tips = True)

    h_meshes = rh_model.hand_meshes(output)
    j_meshes = rh_model.joint_meshes(output)

    # visualize hand mesh only
    # h_meshes[0].show()
    h_meshes[0].export('./h_meshes.ply')

    # visualize joints mesh only
    # j_meshes[0].show()
    j_meshes[0].export('./j_meshes.ply')

    # visualize hand and joint meshes
    hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])
    # hj_meshes.show() 
    hj_meshes.export('./hj_meshes.ply')