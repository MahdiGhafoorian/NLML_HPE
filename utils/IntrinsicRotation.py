# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:39:59 2024

@author: Usuario
"""

import numpy as np
import torch

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def radian(angles_degrees):
    angles_radians = angles_degrees * (torch.pi / 180)
    return angles_radians

# Define the rotation matrices using PyTorch
def rotation_matrix_x(theta_x):
    return torch.tensor([
        [1, 0, 0],
        [0, torch.cos(theta_x), -torch.sin(theta_x)],
        [0, torch.sin(theta_x), torch.cos(theta_x)]
    ])

def rotation_matrix_y(theta_y):
    return torch.tensor([
        [torch.cos(theta_y), 0, torch.sin(theta_y)],
        [0, 1, 0],
        [-torch.sin(theta_y), 0, torch.cos(theta_y)]
    ])

def rotation_matrix_z(theta_z):
    return torch.tensor([
        [torch.cos(theta_z), -torch.sin(theta_z), 0],
        [torch.sin(theta_z), torch.cos(theta_z), 0],
        [0, 0, 1]
    ])

# Function to apply intrinsic rotation
def intrinsic_rotation(point, theta_x, theta_y, theta_z):
    # Compute the individual rotation matrices
    Rx = rotation_matrix_x(theta_x)
    Ry = rotation_matrix_y(theta_y)
    Rz = rotation_matrix_z(theta_z)
    
    R = Rx @ Ry @ Rz  # applying BIWI rotation convention
    # R = R.T           # applying BIWI rotation convention
    
    # Apply the rotations 
    if point is not None:
        # rotated_point = Ry @ Rx @ point @ Rz # applying my rotation convention
        rotated_point = torch.matmul(R, point) # applying BIWI rotation convention
    else: 
        # rotated_point = Ry @ Rx @ Rz  # applying my rotation convention
        rotated_point = Rx @ Ry @ Rz  # applying BIWI rotation convention
        
    
        
    rotated_point = rotated_point.to(device)
    
    return rotated_point
    
def get_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    # Convert degree to radians
    theta_x = radian(torch.tensor(pitch_deg * -1))  # Rotation around x-axis 
    theta_y = radian(torch.tensor(yaw_deg)) # Rotation around y-axis 
    theta_z = radian(torch.tensor(roll_deg))  # Rotation around z-axis 
    
    # Manually computed intrinsic rotation, multiply the rotation matrices in the reverse order
    rotation_matrix = intrinsic_rotation(None, theta_x, theta_y, theta_z)      
    print("Rotation martix:", rotation_matrix)
    return rotation_matrix

def apply_rotation_to_points(landmark_points, yaw_deg, pitch_deg, roll_deg):
    
    # Convert degree to radians
    theta_x = radian(torch.tensor(pitch_deg * -1))  # Rotation around x-axis 
    theta_y = radian(torch.tensor(yaw_deg)) # Rotation around y-axis 
    theta_z = radian(torch.tensor(roll_deg))  # Rotation around z-axis 
    
    # Apply rotation to all points
    rotated_points = torch.zeros_like(landmark_points)
    for i, point in enumerate(landmark_points):
        rotated_points[i] = intrinsic_rotation(point, theta_x, theta_y, theta_z)
    return rotated_points
        

# Function to create rotation matrices
def rotation_matrix(axis, angle):
    angle_rad = np.radians(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(angle_rad), -np.sin(angle_rad)],
                         [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        return np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                         [0, 1, 0],
                         [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        return np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                         [np.sin(angle_rad), np.cos(angle_rad), 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

# Function to apply rotations based on the selected type
def apply_rotations(landmarks, angles, rotation_type='extrinsic'):
    rot_x = rotation_matrix('x', angles[1])
    rot_y = rotation_matrix('y', angles[0])
    rot_z = rotation_matrix('z', angles[2])

    if rotation_type == 'extrinsic':
        # Apply rotations: First rotate around z, then y, then x
        return landmarks @ rot_z.T @ rot_y.T @ rot_x.T
    elif rotation_type == 'intrinsic':
        # Apply intrinsic rotations: Rotate around x, then y, then z
        return landmarks @ rot_x @ rot_y @ rot_z
    else:
        raise ValueError("Rotation type must be 'intrinsic' or 'extrinsic'.")
