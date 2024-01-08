from __future__ import print_function, division
import argparse
import torch
import os,sys
import numpy as np
import joblib
import smplx
import trimesh
import h5py

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from smplify import SMPLify3D
import config

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1,
                        help='input batch size')
    parser.add_argument('--num_smplify_iters', type=int, default=100,
                        help='num of smplify iters')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables cuda')
    parser.add_argument('--gpu_ids', type=int, default=0,
                        help='choose gpu ids')
    parser.add_argument('--num_joints', type=int, default=22,
                        help='joint number')
    parser.add_argument('--joint_category', type=str, default="AMASS",
                        help='use correspondence')
    parser.add_argument('--fix_foot', type=str, default="False",
                        help='fix foot or not')
    parser.add_argument('--data_folder', type=str, default="./demo/demo_data/",
                        help='data in the folder')
    parser.add_argument('--save_folder', type=str, default="./demo/demo_results/",
                        help='results save folder')
    parser.add_argument('--files', type=str, default="test_motion.npy",
                        help='files use')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()
    print("opt: ", opt)

    device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu") # device
    print("SMPL_MODEL_DIR: ", config.SMPL_MODEL_DIR)

    smplmodel = smplx.create(config.SMPL_MODEL_DIR, model_type = "smpl", gender = "neutral", ext = "pkl", batch_size = opt.batchSize).to(device)

    # load the mean pose as original
    smpl_mean_file = config.SMPL_MEAN_FILE
    print("load the mean pose as original: ", smpl_mean_file)
    file = h5py.File(smpl_mean_file, 'r')
    init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float() # [1 72]
    init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float() # [1 10]
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device) # [3]

    pred_pose = torch.zeros(opt.batchSize, 72).to(device) # [1 72]
    pred_betas = torch.zeros(opt.batchSize, 10).to(device) # [1 10]
    pred_cam_t = torch.zeros(opt.batchSize, 3).to(device) # [1 3]
    keypoints_3d = torch.zeros(opt.batchSize, opt.num_joints, 3).to(device) # [1 22 3]

    # initialize SMPLify
    smplify = SMPLify3D(smplxmodel=smplmodel, batch_size=opt.batchSize,
                    joints_category=opt.joint_category, num_iters=opt.num_smplify_iters, device=device)
    print("initialize SMPLify3D done!")

    # load data
    purename = os.path.splitext(opt.files)[0] # test_motion
    data = np.load(opt.data_folder + "/" + purename + ".npy") # './demo/demo_data//test_motion.npy'
    dir_save = os.path.join(opt.save_folder, purename) # './demo/demo_results/test_motion'
    if not os.path.isdir(dir_save):
        os.makedirs(dir_save, exist_ok=True) 
    
    # run the whole seqs
    num_seqs = data.shape[0]
    for idx in range(num_seqs):
        print(f"frame_idx = {idx}")
        joints3d = data[idx] # frame data [22, 3]
        keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(device).float() # 当前帧的真实3D关节旋转坐标

        if idx == 0: # 第一帧使用平均的betas
            pred_betas[0, :] = init_mean_shape
            pred_pose[0, :] = init_mean_pose
            pred_cam_t[0, :] = cam_trans_zero
        else: # 后续帧使用前一帧保存的betas
            data_param = joblib.load(dir_save + "/" + "%04d"%(idx-1) + ".pkl")
            pred_betas[0, :] = torch.from_numpy(data_param['beta']).unsqueeze(0).float() # [1 1 10]
            pred_pose[0, :] = torch.from_numpy(data_param['pose']).unsqueeze(0).float() # [1 1 72]
            pred_cam_t[0, :] = torch.from_numpy(data_param['cam']).unsqueeze(0).float() # [1 1 3]

        if opt.joint_category == "AMASS":
            confidence_input = torch.ones(opt.num_joints) # [22]
            # make sure the foot and ankle
            if opt.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        # from initial to fitting # 利用SMPLify模型，基于真实的3D关节旋转坐标来估计betas等参数，重建3维人体
        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = smplify(
            pred_pose.detach(), pred_betas.detach(), pred_cam_t.detach(),
            keypoints_3d, conf_3d=confidence_input.to(device), seq_ind=idx)
        
        # save the results to ply
        outputp = smplmodel(betas=new_opt_betas, global_orient=new_opt_pose[:, :3], body_pose=new_opt_pose[:, 3:],
                    transl=new_opt_cam_t, return_verts=True)    
        mesh_p = trimesh.Trimesh(vertices=outputp.vertices.detach().cpu().numpy().squeeze(), faces=smplmodel.faces, process=False)
        mesh_p.export(dir_save + "/" + "%04d"%idx + ".ply")

        # save the pkl
        param = {}
        param['beta'] = new_opt_betas.detach().cpu().numpy() # [1 10]
        param['pose'] = new_opt_pose.detach().cpu().numpy() # [1 72]
        param['cam'] = new_opt_cam_t.detach().cpu().numpy() # [1 3]

        # save the root position
	    # shape of keypoints_3d is torch.Size([1, 22, 3]) and root is the first one
        root_position = keypoints_3d[0, 0, :].detach().cpu().numpy()
        print(f"root at {root_position}, shape of keypoints_3d is {keypoints_3d.shape}")
        param['root'] = root_position
        joblib.dump(param, dir_save + "/" + "%04d"%idx + ".pkl", compress=3)
    print("All done!")