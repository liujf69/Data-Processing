# Download dataset
Download UAV-Human Dataset from [Here](https://sutdcv.github.io/uav-human-web/)
```
- Skeleton/
  - P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
  - P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
  ...
```

# Process dataset
1. Visualize dataset
```
python vis_pose.py
```
You can view the sample visualization in vis_imgs. <br />

2. Statistics dataset
```
python statistics_samples.py
```
You can count all samples and maximum frames.  <br />

3. Split dataset
```
python extract_pose.py
```
You can get training and testing data for Cross-Subject-v1 and Cross-Subject-v2. <br />

4. Load dataset
```
python feeder.py
```
You can simply load the processed data. <br />

# Contact
For any questions, feel free to contact: ```liujf69@mail2.sysu.edu.cn```

