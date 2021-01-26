## __VesselSeg-Pytorch__ : _Retinal vessel segmentation toolkit based on pytorch_
### Introduction
This project is a retinal blood vessel segmentation code based on python and pytorch framework, including data preprocessing, model training and testing, visualization, etc. This project is suitable for researchers who study retinal vessel segmentation.  
![Segmentation results](http://ww1.sinaimg.cn/mw690/a2d5ce76ly1gn16jnugz6j20ri08w465.jpg)  
### Requirements  
The main package and version of the python environment are as follows
```
# Name                    Version         
python                    3.7.9                    
pytorch                   1.7.0         
torchvision               0.8.0         
cudatoolkit               10.2.89       
cudnn                     7.6.5           
matplotlib                3.3.2              
numpy                     1.19.2        
opencv                    3.4.2         
pandas                    1.1.3        
pillow                    8.0.1         
scikit-learn              0.23.2          
scipy                     1.5.2           
tensorboardX              2.1        
tqdm                      4.54.1             
```  
The above environment is successful when running the code of the project. In addition, it is well known that pytorch has very good compatibility (version>=1.0). Thus, __I suggest you try to use the existing pytorch environment firstly.__ 

---  
## Usage 
### 0) Download Project 

Running```git clone https://github.com/lee-zq/VesselSeg-Pytorch.git```  
The project structure and intention are as follows : 
```
VesselSeg-Pytorch			# Source code		
    ├── config.py		 	# Configuration information
    ├── lib			            # Function library
    │   ├── common.py
    │   ├── dataset.py		        # Dataset class to load training data
    │   ├── extract_patches.py		# Extract training and test samples
    │   ├── help_functions.py		# 
    │   ├── __init__.py
    │   ├── logger.py 		        # To create log
    │   ├── losses
    │   ├── metrics.py		        # Evaluation metrics
    │   ├── pre_processing.py		# Data preprocessing
    ├── models		        # All models are created in this folder
    │   ├── denseunet.py
    │   ├── __init__.py
    │   ├── LadderNet.py
    │   ├── nn
    │   └── UNetFamily.py
    ├── prepare_dataset	        # Prepare the dataset (organize the image path of the dataset)
    │   ├── chasedb1.py
    │   ├── data_path_list		# image path of dataset
    │   ├── drive.py
    │   └── stare.py
    ├── test.py			            # Test file
    ├── tools			     # some tools
    │   ├── ablation_plot.py
    │   ├── ablation_plot_with_detail.py
    │   ├── merge_k-flod_plot.py
    │   └── visualization
    └── train.py			        # Train file
```
### 1) Datasets preparation 
1. Please download the retina image datasets(DRIVE, STARE and CHASE_DB1) from [TianYi Cloud](https://cloud.189.cn/t/UJrmYrFZBzIn). Otherwise, you can download three data sets from the official address: [DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/),[STARE](http://www.ces.clemson.edu/ahoover/stare/) and [CHASE_DB1]().  
2. Unzip the downloaded `datasets.rar` file. The results are as follows:  
```
  datasets
    ├── CHASEDB1
    │   ├── 1st_label
    │   ├── 2nd_label
    │   ├── images
    │   └── mask
    ├── DRIVE
    │   ├── test
    │   └── training
    └── STARE
        ├── 1st_labels_ah
        ├── images
        ├── mask
        └── snd_label_vk
```
3. Create data path index file(.txt). running:
Please modify the data folder path:`data_root_path`(in the [`drive.py`](https://github.com/lee-zq/VesselSeg-Pytorch/blob/master/prepare_dataset/drive.py), [`stare.py`](https://github.com/lee-zq/VesselSeg-Pytorch/blob/master/prepare_dataset/stare.py) and [`chasedb1.py`](https://github.com/lee-zq/VesselSeg-Pytorch/blob/master/prepare_dataset/chasedb1.py)) to the absolute path of the datasets downloaded above  
```
python ./prepare_dataset/drive.py           
```
In the same way, the data path files of the three datasets can be obtained, and the results are saved in the [`./prepare_dataset/data_path_list`](https://github.com/lee-zq/VesselSeg-Pytorch/tree/master/prepare_dataset/data_path_list) folder
### 2) Training model
Please confirm the configuration information in the [`config.py`](https://github.com/lee-zq/VesselSeg-Pytorch/blob/master/config.py). Pay special attention to the `train_data_path_list` and `test_data_path_list`. Then, running:
```
CUDA_VISIBLE_DEVICES=1 python train.py --save UNet_vessel_seg --batch_size 64
```
You can configure the training information in config, or modify the configuration parameters using the command line. The training results will be saved to the corresponding directory(save name) in the `experiments` folder.  
### 3) Testing model
The test process also needs to specify parameters in [`config.py`](https://github.com/lee-zq/VesselSeg-Pytorch/blob/master/config.py). You can also modify the parameters through the command line, running:
```
CUDA_VISIBLE_DEVICES=1 python test.py --save UNet_vessel_seg  
```  
The above command loads the `best_model.pth` in `./experiments/UNet_vessel_seg` and performs a performance test on the testset, and its test results are saved in the same folder.    

## Visualization of results
0. Training sample visualization  
![train data](http://ww1.sinaimg.cn/mw690/a2d5ce76ly1gn1710u3s4j20hg06y0tt.jpg)  
1. Segmentation results (the original image, predicted probability image, predicted binary image and groundtruth)  
![results drive3](http://ww1.sinaimg.cn/mw690/a2d5ce76ly1gn16yw1u1qj21qs0g8tou.jpg)   
![results drive6](http://ww1.sinaimg.cn/mw690/a2d5ce76ly1gn16zbiob1j21qs0g8qlw.jpg)  
![results drive6](http://ww1.sinaimg.cn/mw690/a2d5ce76ly1gn16zbiob1j21qs0g8qlw.jpg)  
2. ROC Curve and PR Curve
![roc drive6](http://ww1.sinaimg.cn/mw690/a2d5ce76ly1gn1740h9h1j20hs0dct9c.jpg)![pr drive6](http://ww1.sinaimg.cn/mw690/a2d5ce76ly1gn174q0dguj20hs0dcaam.jpg)  
## Others