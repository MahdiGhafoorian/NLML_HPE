# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:37:51 2024

Code to Create Tensor for Manifold Learning

@author: Mahdi Ghafourian
"""

# Standard library
import os
import glob

# Third-party libraries
import torch
import mediapipe as mp
import tensorly as tl

# Local application imports
from helpers import IntrinsicRotation as IR
from helpers import FeatureExtractor as FE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def discretize_interval(a, b, n):
    """
    Discretize the interval [a, b] into n equally spaced points.

    Parameters:
    a (float): The start of the interval.
    b (float): The end of the interval.
    n (int): The number of points to discretize into.

    Returns:
    list of float: A list of n equally spaced points in the interval [a, b].
    """
    if n <= 0:
        raise ValueError("Number of points must be positive.")
    if a >= b:
        raise ValueError("Start of interval must be less than end of interval.")
        
     # Calculate spacing between points
    spacing = (b - a) / (n - 1)

    # Generate equally spaced points
    points = [a + i * spacing for i in range(n)]
    return points, spacing


# This function creates the tensor for the decomposition and filling it 
def Compose_Tensor(input_image_path, tensor_shape, yaw_discretized, 
                       pitch_discretized, roll_discretized, identities, use_rotation_features):
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)    
    
    # Initialize the tensor to store filenames
    composition_tensor = torch.zeros(tensor_shape)
    base_landmarks = [None] * len(identities)
    not_detected_cnt = [0] * len(identities)

    # iterating over bins to fill the tensor
    for yaw_idx, yaw_val in enumerate(yaw_discretized):
        for pitch_idx, pitch_val in enumerate(pitch_discretized):
            for roll_idx, roll_val in enumerate(roll_discretized):
                
                # iterating over identities with close bins of euler angles
                for id_idx, identity in enumerate(identities):
                    image_path = os.path.join(input_image_path, identity)
                    
                    # get the landmark of (0,0,0) for the given identity as the base landmark
                    if base_landmarks[id_idx] is None:
                        base_img_path = os.path.join(image_path, f'ID{identity}_(0_0_0)')
                        base_img_path = glob.glob(base_img_path + ".*")[0]
                        landmarks_tensor = FE.get_feature_vector(face_mesh, base_img_path, normalize=True)
                        base_landmarks[id_idx]  = torch.tensor(landmarks_tensor).reshape(468, 3)         
                    
                    # If we use landmark detector to extract features
                    if(use_rotation_features == 0):                    
                        cur_img_path = os.path.join(image_path, f'ID{identity}_({yaw_val}_{pitch_val}_{roll_val})')
                        cur_img_path = glob.glob(cur_img_path + ".*")[0]
                        features_vector = FE.get_feature_vector(face_mesh, cur_img_path, normalize=True)
                                                
                        # Check if any landmark is detected, if all zero means not detected
                        all_zero = torch.all(features_vector == 0)
                        
                        # if no landmark detected on the given pose, we compute the corresponing feature 
                        # vector by rotating the base landmark (landmark at x=0,y=0,z=0) with the given pose                    
                        if all_zero:
                            not_detected_cnt[id_idx] += 1
                            features_vector = IR.apply_rotation_to_points(base_landmarks[id_idx], yaw_val, pitch_val, roll_val)
                            features_vector = features_vector.view(-1)
                            # features_vector = features_vector[:, :2].reshape(-1)
                    # If we use rotation matrix of the to get the features at the given pose        
                    elif(use_rotation_features == 1):  
                        if(yaw_val == 0 and pitch_val ==0 and roll_val==0):
                            features_vector = base_landmarks[id_idx]
                        else:           
                            # Define rotation angles for x, y, z axes
                            angles = [yaw_val, pitch_val, roll_val]  # [angle_x, angle_y, angle_z]

                            # Choose rotation type: 'extrinsic' or 'intrinsic'
                            rotation_type = 'intrinsic'  # Change this to 'intrinsic' or 'extrinsic' if needed

                            # Apply rotations based on selected type
                            rotated_landmarks = IR.apply_rotations(base_landmarks[id_idx].numpy(), angles, rotation_type)

                            # Convert back to PyTorch tensor
                            features_vector = torch.tensor(rotated_landmarks, dtype=torch.float32)
                            # features_vector = IR.apply_rotation_to_points(base_landmarks[id_idx], yaw_val, pitch_val, roll_val)
                        features_vector = features_vector.view(-1) # flatten features as one vector
                    
                    composition_tensor[id_idx, yaw_idx, pitch_idx, roll_idx, :] = features_vector
        print(f"All Ids for Yaw {yaw_val} are recorded!")                                        
    composition_tensor = tl.tensor(composition_tensor)
    print("Not detected idenities = ")
    print(not_detected_cnt)
    return composition_tensor, False


