# Extract_NTU_Person
This is a code support for the **IPP-Net**、**EPP-Net**. <br />
You can generate the image frame required by IPP-Net、EPP-Net as follows.
![image](https://github.com/liujf69/Extract_NTU_Person/blob/master/Fig.png)

# Prepare datasets:
1. Download **NTU RGB+D** Video dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/) <br />
2. Extract and put downloaded data into the following directory structure.
```
- Video/
  - NTU60/
    S001C001P001R001A001_rgb.avi
    S001C001P001R001A002_rgb.avi
    ...
  - NTU120/
    S018C001P008R001A061_rgb.avi
    S018C001P008R001A062_rgb.avi
    ...
```
# Extract video frame:
**Run:** 
```
python Extract_frame.py --sample_name_path sample_name_path \
    --video_path video_path \
    --output_path output_path
```
**Example:** 
```
python Extract_frame.py --sample_name_path ./test_sample.txt \
    --video_path ./Video \
    --output_path ./Video_Frame
```
# Extract person frame:
**Run:** 
```
python Extract_person.py --sample_name_path ./test_sample.txt \
      --frame_path frame_path \
      --output_path output_path \
      --model_path model_path \
      --data_yaml data_yaml
```
**Example:**
```
python Extract_person.py --sample_name_path ./test_sample.txt \
    --frame_path ./Video_Frame \
    --output_path ./Person_Frame \
    --model_path ./pretrained/yolov5m.pt \
    --data_yaml ./pretrained/coco128.yaml
```



