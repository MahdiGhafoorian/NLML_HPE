
# NLML_HPE: Head Pose Estimation with Limited Data via Manifold Learning


This repository is the official implementation of [NLML_HPE](https://github.com/MahdiGhafoorian/NLML_HPE) head pose estimation method.

## Overview

<p align="center">
  <img src="assets/NLML_HPE_demo.gif" />
</p>

We proposed a novel deep learning approach for head pose estimation with limited training data via non-linear manifold learning called NLML-HPE. This method is based on the combination of tensor decomposition (i.e., Tucker decomposition) and feed forward neural network. Unlike traditional classification-based approaches, our method formulates head pose estimation as a regression problem, mapping input landmarks to a continuous representation of pose angles.

##  Preparation

### Environments  
  python == 3.9, torch >= 1.10.1, CUDA ==11.2

### Datasets  

The dataset required to train our model must contain a sample for every combination of Euler angles (yaw, pitch, and roll) for each identity, as our method relies on fully populating all entries of the tensor. To meet this requirement, we rendered all desired combinations of Euler angles from 3D models in FaceScape for each identity using PyTorch3D.

We provide a small subset of this dataset, which is permitted for publication. This subset is intended to help you better understand the structure and format of our training and validation data. The provided dataset has already been rendered by us for the specified angle combinations and can be downloaded at:

[DOWNLOAD LINK]

For full-scale training, you will need to download the complete FaceScape dataset from the following address. You can then render one random 3D model per identity with your desired pose combinations.

[FACEscape DOWNLOAD LINK]

For your validation, follow the [6DRepnet](https://github.com/thohemp/6DRepNet) to prepare the following datasets:

* **300W-LP**, **AFLW2000** from [here](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).

* **BIWI** (Biwi Kinect Head Pose Database) from [here](https://icu.ee.ethz.ch/research/datsets.html). 

Store them in the *datasets* directory.


For 300W-LP and AFLW2000 we need to create a *filenamelist*. 
```
python create_filename_list.py --root_dir datasets/300W_LP
```
The BIWI datasets needs be preprocessed by a face detector to cut out the faces from the images. You can use the script provided [here](https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py). For 7:3 splitting of the BIWI dataset you can use the equivalent script [here](https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi_70_30.py). The cropped image size is set to *256*.

### Directory structure
* After preparation, you will be able to see the following directory structure: 
  ```
  NLML_HPE
  ├── datasets
  │   ├── 300W_LP
  │     ├── files.txt
  │     ├── ...
  │   ├── AFLW2000 
  │     ├── files.txt
  │     ├── ... 
  │   ├── ...
  ├── assets
  ├── create_filename_list.py
  ├── datasets.py
  ├── README.md
  ├── ...
  ```
## Training & Evaluation

To do both training and evaluation, you need first to configure you setting, paths, etc. in all config files found in configs folder. 

To train our model, you need to run the following scripts in the given order: TD_main.py, NLML_HPE_EncoderTrainer.py, NLML_HPE_MLPHeadsTrainer.py, and NLML_HPE_Model_Builder.py.


```sh
python TD_main.py  --use_rotation_features=0 
		   --perform_validation=False 

python NLML_HPE_EncoderTrainer.py

python NLML_HPE_MLPHeadsTrainer.py

python NLML_HPE_Model_Builder.py

```

You can evaluate the model following:

```sh
python NLML_HPE_Test.py
```

## Inference and Test
We have two scripts for the inference. The first one, optimizes the cosine function with the (learned) optimized sinusoidal parameters from TD_main. This inference can be performed by calling TD_Inference. However, it is slow and time-consuming, the reason we trained an encoder plus three MLP heads to train a FFN that learn to do the same inference but in real-time. To do this, we provided NLML_HPE_Test to do the inference on the selected test set. You can use either scripts for the inference on the input image:

```sh
python TD_Inference.py --image_path="{your image}"
```
or
```sh
python NLML_HPE_Test.py
```

## Main results


We provide some results on AFLW2000 with models trained on 300W_LP. These models are trained on one TITAN V GPU. 

|          config          | MAE  | VMAE | training |   download |
|:------------------------:|:----:|:----:|:--------:|:-------------:|
| TokenHPEv1-ViT/B-224*224-lyr3 | 4.81 | 6.09 | ~24hours |   [gdrive](https://drive.google.com/file/d/1bqfJs4mvQd4jQELsj3utEEeS6SDzW30_/view?usp=sharing)     |


## **Acknowledgement**
Many thanks to the authors of [6DRepnet](https://github.com/thohemp/6DRepNet). We reuse their code for data preprocessing and evaluation which greatly reduced redundant work.

## **Citation**

If you find our work useful, please cite the paper:

```
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Cheng and Liu, Hai and Deng, Yongjian and Xie, Bochen and Li, Youfu},
    title     = {TokenHPE: Learning Orientation Tokens for Efficient Head Pose Estimation via Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {8897-8906}
}
```
