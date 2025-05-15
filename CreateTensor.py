# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:37:51 2024

Code to Create Tensor for Manifold Learning

@author: Mahdi
"""

import os
import struct
import torch
import mediapipe as mp
import numpy as np
import tensorly as tl
from utils import IntrinsicRotation as IR


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

def find_corresonding_image(directory, name_part):
    """
    Search for PNG files in the given directory that contain the specified name_part.

    Parameters:
    directory (str): The directory to search in.
    name_part (str): The part of the filename to search for.

    Returns:
    list: List of matching PNG filenames.
    """
    matching_files = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a PNG and contains the name_part
        if filename.endswith('.png') and name_part in filename:
            full_path = os.path.join(directory, filename)
            matching_files.append(full_path)

    return matching_files
        
def read_head_pose_values(bin_file_path):
    # Open the binary file in read mode
    with open(bin_file_path, 'rb') as file:
        # Read the first 12 bytes (3 floats * 4 bytes each = 12 bytes)
        data = file.read(24)
        
        # Unpack the bytes into three float values
        head_pose_values = struct.unpack('ffffff', data)
        
        return head_pose_values[3],head_pose_values[4], head_pose_values[5]

def load_all_pose_files(folder_path):
    """Load all pose files and store their yaw, pitch, roll with filenames."""
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.bin'):
            file_path = os.path.join(folder_path, file_name)
            yaw, pitch, roll = read_head_pose_values(file_path)
            data.append((file_name, yaw, pitch, roll))
    return data


def Compose_Tensor_with_pose_file(input_pose_data_path, input_image_path, tensor_shape, yaw_discretized, 
                       pitch_discretized, roll_discretized, yaw_discretized_spacing, 
                       pitch_discretized_spacing, roll_discretized_spacing, identities):
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)    
    empty_entry = False
    
    """Find closest files based on discretized yaw, pitch, roll values."""
    # Discretization of yaw, pitch, and roll based on tensor shape
    # yaw_discretized, yaw_discretized_spacing = discretize_interval(-1*yaw_range, yaw_range,tensor_shape[0])
    # pitch_discretized, pitch_discretized_spacing = discretize_interval(-1*pitch_range, pitch_range,tensor_shape[1])
    # roll_discretized, roll_discretized_spacing = discretize_interval(-1*roll_range, roll_range,tensor_shape[2])
    

    # Initialize the tensor to store filenames
    closest_files_tensor = np.empty(tensor_shape, dtype=object)
    composition_tensor = torch.zeros(tensor_shape)

    # iterating over bins
    for yaw_idx, yaw_val in enumerate(yaw_discretized):
        for pitch_idx, pitch_val in enumerate(pitch_discretized):
            for roll_idx, roll_val in enumerate(roll_discretized):
                
                # iterating over identities with colse bins of euler angles
                for id_idx, identity in enumerate(identities):
                    pose_data_path = image_path = ''
                    pose_data_path = os.path.join(input_pose_data_path, identity)
                    image_path = os.path.join(input_image_path, identity)
                    # Initialize variables to track the closest file
                    closest_file = None
                    min_distance = float('inf')      
                    
                    # Step 1: Load all .bin files and their yaw, pitch, roll values
                    data = load_all_pose_files(pose_data_path)
                    
                    for file_name, yaw, pitch, roll in data:
                        # print(yaw, pitch, roll)
                        if abs(yaw - yaw_val) <= (yaw_discretized_spacing) and \
                           abs(pitch - pitch_val) <= (pitch_discretized_spacing): #and \
                           #abs(roll - roll_val) <= (roll_discretized_spacing):
                               
                            #Calculate Euclidean distance for closetst discretized values overall                         
                            distance = np.sqrt((yaw - yaw_val)**2 + (pitch - pitch_val)**2 + (roll - roll_val)**2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_file = file_name
                                #///////////////////////// Get the landmarks /////////////////////////////
                                # closest_file[0:11] is the pure file name
                                image_full_path = find_corresonding_image(image_path, closest_file[0:11]) 
                                
                                features_vector = get_feature_vector(face_mesh, image_full_path[0], normalized=True)
                        
                                #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
                    # Store the closest file name in the corresponding index of the tensor
                    if closest_file:
                        # print(f'closest samples: {image_full_path},  {closest_file}')
                        # if(closest_file == 'frame_00479_pose.bin'):
                        #     print(f'yes, yaw = {yaw_idx}, pitch={pitch_idx}, roll={roll_idx}')
                        # closest_files_tensor[yaw_idx, pitch_idx, roll_idx] = closest_file
                        composition_tensor[id_idx, yaw_idx, pitch_idx, roll_idx, :] = features_vector
                    else:
                        empty_entry = True
    composition_tensor = tl.tensor(composition_tensor)
    return composition_tensor, empty_entry


      
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
                        base_img_path = os.path.join(image_path, f'ID{identity}_(0_0_0).png')
                        landmarks_tensor = get_feature_vector(face_mesh, base_img_path, normalized=True)
                        base_landmarks[id_idx]  = torch.tensor(landmarks_tensor).reshape(468, 3)         
                    
                    # If we use landmark detector to extract features
                    if(use_rotation_features == 0):                    
                        cur_img_path = os.path.join(image_path, f'ID{identity}_({yaw_val}_{pitch_val}_{roll_val}).png')
                        features_vector = get_feature_vector(face_mesh, cur_img_path, normalized=True)
                                                
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


