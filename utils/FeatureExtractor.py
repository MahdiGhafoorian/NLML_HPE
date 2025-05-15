# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:32:33 2025

This script provides functions to extract facial features under different input conditions

@author: Mahdi Ghafourian
"""



import cv2
import torch
import numpy as np


def Normalization_using_Centroid(landmark):
    
    landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmark])
    centroid = np.mean(landmark_array, axis=0)            
    centered_landmarks = landmark_array - centroid            
    norm_squared = np.sum(centered_landmarks ** 2, axis=1) 
    mean_norm_squared = np.mean(norm_squared)
    scale_factor = np.sqrt(mean_norm_squared)
    scaled_landmarks = centered_landmarks / scale_factor
    landmark_list = scaled_landmarks.flatten().tolist()
    
    return landmark_list

def Read_Landmarks_and_Normalizing_using_IPD(landmark, ref_list, normalize):
    
    landmark_list = []
    
    # Find interpupillary distance (IPD) for additional scaling
    left_eye_idx = 33  # Left eye corner (MediaPipe index)
    right_eye_idx = 263  # Right eye corner (MediaPipe index)

    left_eye = np.array([landmark[left_eye_idx].x,
                         landmark[left_eye_idx].y,
                         landmark[left_eye_idx].z])

    right_eye = np.array([landmark[right_eye_idx].x,
                          landmark[right_eye_idx].y,
                          landmark[right_eye_idx].z])

    ipd = np.linalg.norm(left_eye - right_eye)
    if ipd == 0:  # Avoid division by zero
        ipd = 1e-6 
                
    for landmarks in landmark:
        x = landmarks.x
        y = landmarks.y
        z = landmarks.z
        if(normalize == True): # subtract ref point from all landmark points for normalizing translation
                                # Devide landmark points by ipd for scale nomalization
            x -= ref_list[0]
            y -= ref_list[1]
            z -= ref_list[2]
            
            x /= ipd
            y /= ipd
            z /= ipd
            
        landmark_list.extend([x,y,z])  
    
    return landmark_list    




def get_feature_vector(face_mesh, full_path, normalize):
    features_vector = []
    image = cv2.imread(full_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    

    # Get the result of face mesh module 
    results = face_mesh.process(image)   

    landmark_list = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ref_point = face_landmarks.landmark[1] # landmark point at nose_tip
            ref_list = [ref_point.x, ref_point.y, ref_point.z]
            
            ############### Read landmarks and Normalize using centroid ###############
          
            # landmark_list = Normalization_using_Centroid(face_landmarks.landmark)
            
            ###########################################################################
            
                            ######### OR ##########
            
            ####################### Read and Normalize using IPD ######################
                      
            landmark_list = Read_Landmarks_and_Normalizing_using_IPD(face_landmarks.landmark, ref_list, normalize)
            
            ###########################################################################

            
            landmarks_tensor = torch.tensor(landmark_list[0:1404]).float() # 468 is the total number of landmarks extracted
            # landmarks_tensor = landmark_list[0:6][0]
            
    # populating features (for the simplicity temporarily recording first landmark x,y,z)          
    if(results.multi_face_landmarks is None):
        features_vector = torch.tensor(1404 * [0]).float()  # no landmark is extracted
        # features_vector = 0 # no landmark is extracted
    else:
        features_vector = landmarks_tensor # landmark_list[0:6]
    
    return features_vector



def get_feature_vector_from_nparray(face_mesh, image, normalize):
    features_vector = []
    
    # Get the result of face mesh module 
    results = face_mesh.process(image)   

    landmark_list = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ref_point = face_landmarks.landmark[1] # landmark point at nose_tip
            ref_list = [ref_point.x, ref_point.y, ref_point.z]
            
            ############### Read landmarks and Normalize using centroid ###############
          
            # landmark_list = Normalization_using_Centroid(face_landmarks.landmark)
            
            ###########################################################################
            
                            ######### OR ##########
            
            ####################### Read and Normalize using IPD ######################
                      
            landmark_list = Read_Landmarks_and_Normalizing_using_IPD(face_landmarks.landmark, ref_list, normalize)
            
            ###########################################################################

            landmarks_tensor = torch.tensor(landmark_list[0:1404]).float()  # 468 is the total number of landmarks extracted
            # landmarks_tensor = landmark_list[0:6][0]
            
    # populating features (for the simplicity temporarily recording first landmark x,y,z)          
    if(results.multi_face_landmarks is None):
        features_vector = torch.tensor(1404 * [0]).float()  # no landmark is extracted
        # features_vector = 0 # no landmark is extracted
    else:
        features_vector = landmarks_tensor # landmark_list[0:6]
    
    return features_vector



def get_feature_vector_from_image(face_mesh, image, normalize, isPil):
    features_vector = []
    if isPil:
        image = np.array(image, dtype=np.uint8)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    # Get the result of face mesh module 
    results = face_mesh.process(image)   

    landmark_list = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ref_point = face_landmarks.landmark[1] # landmark point at nose_tip
            ref_list = [ref_point.x, ref_point.y, ref_point.z]
            
            ############### Read landmarks and Normalize using centroid ###############
          
            # landmark_list = Normalization_using_Centroid(face_landmarks.landmark)
            
            ###########################################################################
            
                            ######### OR ##########
            
            ####################### Read and Normalize using IPD ######################
                      
            landmark_list = Read_Landmarks_and_Normalizing_using_IPD(face_landmarks.landmark, ref_list, normalize)
            
            ###########################################################################

            landmarks_tensor = torch.tensor(landmark_list[0:1404]).float()  # 468 is the total number of landmarks extracted
            # landmarks_tensor = landmark_list[0:6][0]
            
    # populating features (for the simplicity temporarily recording first landmark x,y,z)          
    if(results.multi_face_landmarks is None):
        features_vector = torch.tensor(1404 * [0]).float()  # no landmark is extracted
    else:
        features_vector = landmarks_tensor # landmark_list[0:6]
    
    return features_vector

