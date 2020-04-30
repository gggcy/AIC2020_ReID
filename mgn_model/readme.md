This part contains the source codes of training vehicle reid models including resnet152 with MGN, resnet152 with SAC.

## preparing dataset and pretrained model
Users should define their own dataset in the './reid/datasets'
The data should be put into the folder './data' and pretrained models should be put into './weights/'

## training
Run ```sh run_mgn.sh``` to train vehicle reid model. 

Run ```python ./reid/extract_fea_from_val.py``` or ```extract_feature_val.sh``` to extract reid features of each test image. In our implementation, the average feature of one image and its corresponding flip image is used. Please note again, some paths or network architecture name should be modified before running the code. Finally, pickle files containing reid features of each test image will be obtained.  
