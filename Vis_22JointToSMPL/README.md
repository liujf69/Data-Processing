# Use 22 pose joints to visualize SMPL 3D map
# Run
1. Put your pose data (22 joints) into ```./demo/demo_data/```
2. Run ```python fit_seq_ljf.py --files [your_pose_data.npy]```

# Visualization
Origin pose data (22 joints): 
<div align=center>
<img src="./22joints.png" width="400"/>
</div>
Generated SMPL map:
<div align=center>
<img src="./vis_smpl.png" width="600"/>
</div>

# Reference
Our code is based on [joints2smpl](https://github.com/wangsen1312/joints2smpl)
