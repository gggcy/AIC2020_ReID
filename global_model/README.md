This folder contains source codes of training multiple vehicle reid models including se_resnext101, se_resnet152, resnet152, hrnet_48w, se_resnet152_ibnb, densenet161, dpn107, senet154 etc. These codes are based on [open-reid](https://github.com/Cysu/open-reid) and [AICity](https://github.com/wzgwzg/AICity).


**Before running these codes, the environments shouled be prepared and related paths should be modified.**
**Specifically, all dependencies should be installed, the datasets should be put into './data' and pretrained models should be put into './pretrain_models/'.**

## Code Structure 
Move ```experiments/*.sh``` to ```./global_model``` 

Run ```sh train_reid_model.sh``` to train vehicle reid models. To train different models mentioned above, the network architecture name should be modified. Available network names can be found in 'reid/models/\__init\__.py'.  

Run ```sh train_reid_model_sgd_2grafting.sh``` to train vehicle reid models with 2 models grafting. Modify parameters to adjust grafting multiple models

**Notes:** ```train_with_grafting.py``` need to modify line 75 - line 78 according to python version.

Run ```python ./reid/extract_fea_from_val.py``` or ```extract_feature_val.sh``` to extract reid features of each test image. In our implementation, the average feature of one image and its corresponding flip image is used. Please note again, some paths or network architecture name should be modified before running the code. Finally, pickle files containing reid features of each test image will be obtained.  







