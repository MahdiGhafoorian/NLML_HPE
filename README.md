
# NLML_HPE: Head Pose Estimation with Limited Data via Manifold Learning


This repository is the official implementation of [NLML_HPE](https://github.com/MahdiGhafoorian/NLML_HPE) head pose estimation method.

## Overview


<p align="center">
  <img src="assets/NLML_HPE_demo.gif" height="300"/>
  <img src="assets/Girl.gif" height="300"/>
  <img src="assets/Man_call_smartphone.gif" height="300"/>
</p>

We proposed a novel deep learning approach for head pose estimation with limited training data via non-linear manifold learning called NLML-HPE. Our method is based on the combination of tensor decomposition (i.e., Tucker decomposition) and feed forward neural network. Unlike traditional classification-based approaches, our method formulates head pose estimation as a regression problem, mapping input landmarks to a continuous representation of pose angles.

##  Preparation

### Environments  
  python == 3.9, torch >= 1.10.1, CUDA ==11.2

### Datasets  

The dataset required to train our model must contain exactly one sample per every combination of Euler angles (yaw, pitch, and roll) for each identity. This is because our method relies on fully populating all entries of the tensor before decomposition. To meet this requirement, we rendered all desired combinations of Euler angles from 3D models in FaceScape for each identity using PyTorch3D.

We provide a small subset of the our dataset, rendered from the samples that the Facescape authors have permitted to be published. This subset is intended to help you better understand the structure and format of our training and validation data. The provided dataset has already been rendered by us for the specified angle combinations and can be downloaded at:

[DOWNLOAD LINK]

For full-scale training, you will need to download the complete FaceScape dataset from the following address. You can then render one random 3D model per identity with your desired pose combinations.

[FACEscape DOWNLOAD LINK]

For the validation, follow the [6DRepnet](https://github.com/thohemp/6DRepNet) to prepare the following datasets:

* **300W-LP**, **AFLW2000** from [here](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).

* **BIWI** (Biwi Kinect Head Pose Database) from [here](https://icu.ee.ethz.ch/research/datsets.html). 

Store them in the *datasets* directory.


For 300W-LP and AFLW2000 we need to create a *filenamelist*. 
```
python create_filename_list.py --root_dir datasets/300W_LP
```
The BIWI datasets needs be preprocessed by a face detector to cut out the faces from the images. You can use the script provided [here](https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py). The cropped image size is set to *256*.

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

## Inference and Demo
We have two scripts for the inference. The first one, optimizes the cosine function with the (learned) optimized sinusoidal parameters from TD_main. This inference can be performed by calling TD_Inference. However, it is slow and time-consuming, the reason we trained an encoder plus three MLP heads to train a FFN that learn to do the same inference but in real-time. To do this, we provided NLML_HPE_Test to do the inference on the selected test set. You can use either scripts for the inference on the input image:

```sh
python TD_Inference.py --image_path="{your image}"
```
or (better)
```sh
python NLML_HPE_Test.py
```
You can use the following commands to run the demo:

```sh
python generatePose_on_video.py --source = "{webcam or the path to your video}"
		   		--save_output = "{tr for True or fl for False}"
				--output_path = "{the path to save pose included video}"
```

Below are some example videos that were processed by our HPE method, where the estimated head poses are visualized. The source of these videos include my own recordings as well as free for use videos under [Pexels](https://www.pexels.com/) and [Pixabay](https://pixabay.com/) content licence.

## Main results


We provide some results on AFLW2000 with models trained on 300W_LP. These models are trained on one TITAN V GPU. 

| Method       | AFLW2000 Yaw | Pitch | Roll | MAE  | BIWI Yaw | Pitch | Roll | MAE  | AFLW2000 Left | Down | Front | MAEV | BIWI Left | Down | Front | MAEV |
|--------------|--------------|-------|------|------|----------|-------|------|------|----------------|------|-------|------|-----------|------|-------|------|
| 3DDFA	       | 4.71         | 27.08 | 28.43| 20.07| 5.50     |41.90  |13.22 |20.20 | 29.38          |45.00 |35.12  |34.47 | 23.31     |45.00 |35.12  |34.47 |
| Dlib         | 8.50         | 11.25 |22.83 |14.19 |11.66     | 8.22  |19.56 |12.00 | 26.56          |28.51 |14.31  |23.18 | 24.84     |21.70 |14.30  |20.28 |
| HopeNet      | 5.31         | 7.12  | 6.13 | 6.19 | 6.00     | 5.88  | 3.72 | 5.20 | 7.07           | 5.98 | 7.50  | 6.85 | 7.65      | 6.73 | 8.68  | 7.69 |
| FSA-Net      | 4.96         | 6.34  | 4.77 | 5.36 | 4.96     | 4.51  | 3.50 | 4.32 | 6.75           | 6.21 | 7.34  | 6.77 | 6.03      | 5.96 | 7.22  | 6.40 |
| QuatNet      | 3.97         | 5.62  | 3.92 | 4.50 | 2.94     | 4.00  | 4.01 | 3.65 | -              | -    | -     | -    | -         | -    | -     | -    |
| HPE          | 4.80         | 6.18  | 4.87 | 5.28 | 5.12     | 4.02  | 4.57 | 4.57 | -              | -    | -     | -    | -         | -    | -     | -    |
| TriNet       | 4.36         | 5.81  | 4.51 | 4.89 | 4.18     | 5.00  | 4.50 | 4.56 | 6.16           | 5.95 | 6.82  | 6.31 | 5.58      | 5.80 | 7.55  | 6.64 |
| TokenHPE     | **2.68**     | 3.41  |**1.59**|**2.56**|**2.41**| 3.05 |**1.79**|**2.41**|**3.38**|**3.90**|**4.63**|**3.97**|**5.21**|**5.71**|**7.06**|**6.00**|
| 6DRepNet     | 2.79         |**3.39**| 1.65| 2.61 | 3.44     |**2.87**| 2.41 | 2.91 | 3.47           | 3.87 | 4.71  | 4.02 |**4.77**  | 5.12 |**6.48**|**5.65**|
| NLML-HPE (ours)|3.06        | 4.23  | 1.96 | 3.08 | 3.58     | 5.29  | 2.67 | 3.85 | 4.02           | 4.77 | 5.53  | 4.78 | 5.34      | 6.33 | 6.00  | 6.00 |


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
