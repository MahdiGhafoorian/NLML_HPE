# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:51:03 2024

main program of Tensor decomposition for 3D head pose estimation

@author: Mahdi
"""


# Basic required packages
import torch
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import random
import re
import matplotlib.pyplot as plt
import warnings
import scipy.io

# our modules
import CreateTensor
import TD_Trainer
import TD_Tester
from Tester import w_y_values, w_p_values, w_r_values, u_id_values, objective_values
from utils import FeatureExtractor as FE
import EulerAngles_mediapipe


# packages to work with Tucker decomposition
import tensorly as tl
from tensorly import unfold
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor
from numpy.linalg import svd


def main():

    warnings.filterwarnings("ignore")    
    
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ################################ Step 1: Creating Tensor ##########################################
    
    tensor_shape = (300, 11, 9, 7, 1404)   # shape of Tensor (I_1, I_2, ... , I_N)
    tensor_shape = (5, 31, 27, 25, 1404)   # shape of Tensor (I_1, I_2, ... , I_N)
    tensor_shape = (1620, 11, 9, 7, 1404)   # shape of Tensor (I_1, I_2, ... , I_N)
    tensor_shape = (240, 11, 9, 7, 1404)   # shape of Tensor (I_1, I_2, ... , I_N)
    
    
    yaw_bins = np.arange(-50, 51, 10)
    pitch_bins = np.arange(-40, 41, 10)
    roll_bins = np.arange(-30, 31, 10)
    

    input_image_path = "D:/datasets/BIWI/faces_0"
    input_image_path = "3D_DB_(50_40_30)_trainset_singleExpressions_1620_Subjects"
    # path to our generated dataset
    input_image_path = "E:/Mahdi/Databases/3D_DB_(50_40_30)_trainset_singleExpressions_240_Subjects_BIWI_3rd_rotation_convention"
    
    
    # Idenitites that we generated image for using facescape 3D models
    identities = [name for name in os.listdir(input_image_path) if os.path.isdir(os.path.join(input_image_path, name))]
    identities.sort(key=int)
    #----------------------------------------------------- Args --------------------------------------------------
    use_rotation_features = 0  # 0 mean use landmark to create the tensor  |  1 means use rotation matrix to create the tensor
    test_mode = 2 # 0: no teest | 1: getting the Euler angles of a single input image  |  2: compute MAE for the given validation set
    print_singular_values = False # (Debug uses) This flag is used to decide printing singular values of the unfolded tensor 
    plot_factor_matrices_dims = True # (Debug uses) ploting columns of factor matrices
    #-------------------------------------------------------------------------------------------------------------
    
    if use_rotation_features == 0:
        # tensor_file = 'train_tensor_landmarks_normalized_240Subjects_mixedEXP.npy'
        # tensor_file = 'train_tensor_landmarks_normalized_1620Subjects.npy'
        tensor_file = 'train_tensor_landmarks_normalized_240Subjects_BIWI_Convention_3.npy'
        # tensor_file = 'train_tensor_landmarks_normalized_240Subjects_mixedEXP.npy'
    else:
        tensor_file = 'train_tensor_rotations_300Subjects.npy'
        # tensor_file = 'train_tensor_rotations_sase_mini.npy'
    
    # Check if the tensor file exists
    if os.path.exists(tensor_file):
        start_time = time.time()    
        
        # Load the tensor back from disk
        composition_tensor = tl.tensor(np.load(tensor_file))
        
        end_time = time.time()  # Record end time    
        execution_time = end_time - start_time
        print("Tensor loaded from disk.")
        print(f"Time taken: {execution_time} seconds")
    else:
        print("Tensor file does not exist.")
        
        start_time = time.time()  # Record start time   

        
        # empty_entry is a bool telling that if there is an empty cell in the composition tensor
        composition_tensor, empty_entry = CreateTensor.Compose_Tensor(input_image_path, tensor_shape,
                                                                yaw_bins, pitch_bins, roll_bins, identities, use_rotation_features)
        # Save the tensor to disk
        np.save(tensor_file, tl.to_numpy(composition_tensor))  # save the filled tensor to the disk
        
        end_time = time.time()  # Record end time    
        execution_time = end_time - start_time
        print(f"Tensor saved to {tensor_file}.")
        print(f"Time taken: {execution_time} seconds")
   

    ################################ Step 2: Decomposing Tensor #####################################
    start_time = time.time()  # Record start time  
    
    core, factors = tucker(composition_tensor, rank=[5, 3, 3, 3, 1404]) # shape of core (R_1, R_2, ... , R_N)
    
    #----------------------------------- Monitoring block -----------------------------------
    #-------------------- compute and print singular values ---------------------------------
    
    if print_singular_values:
    
        matrix_name = ['U_id', 'yaw', 'pitch', 'roll']
        # Loop through each mode, unfold the tensor, and apply SVD
        for mode in [0,1,2,3]:
            # Unfold the tensor along the current mode
            unfolded_tensor = unfold(composition_tensor, mode)
            
            # Perform SVD on the unfolded matrix
            U, singular_values, Vt = svd(unfolded_tensor, full_matrices=False)
            
            # Print the singular values for the current mode
            # print(f"Singular values for mode {mode}: {singular_values}")
            
            # Calculate the total sum of singular values
            total_sum = np.sum(singular_values)
        
            # Calculate the percentage contribution of each singular value
            percentage_contributions = (singular_values / total_sum) * 100
            
            # Print energies
            for i, contribution in enumerate(percentage_contributions):
                print(f"Singular value {i+1} of {matrix_name[mode]}: {singular_values[i]}, Energy: {contribution:.2f}%")
            print("\n")
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    end_time = time.time() 
    execution_time = end_time - start_time
    print("Tensor successfully decomposed.")
    print(f"Time taken: {execution_time} seconds")
        
    start_time = time.time()  
    # Reconstruct the tensor
    reconstructed_tensor = tucker_to_tensor((core, factors))
    
    # Calculate reconstruction error
    reconstruction_error = tl.norm(composition_tensor - reconstructed_tensor, 2) / tl.norm(composition_tensor, 2)
    print(f'Reconstruction Error: {reconstruction_error}')
    end_time = time.time()  # Record end time    
    execution_time = end_time - start_time
    print(f"Time taken: {execution_time} seconds")
    
    # Extract the core tensor and feature matrix for mode 5
    core_tensor = core
    feature_matrix = factors[4] # shape (I_4, R_4)
    
    # Transpose feature matrix for proper multiplication
    feature_matrix_transposed = tl.transpose(feature_matrix)
    
    # Compute W = core_tensor x_5 factor5
    W = tl.tensordot(core_tensor, feature_matrix_transposed, axes=(4, 0))
    
    print("Shape of W:", W.shape)
    
    ################################ Step 3: Train #####################################
    
    ''' In each matrix of U^(*) where * ∈ {yaw, pitch, role}, 
         rows are discretized angles and columns are ranks or dimensions
    '''
    start_time = time.time()  
    
    yaw_params = (factors[1], yaw_bins)
    pitch_params = (factors[2], pitch_bins)
    roll_params = (factors[3], roll_bins)
    
    # These optimized values are the trignometric parameters that are optimized so the cosine curve fits very well for each dimension
    optimized_yaw, optimized_pitch, optimized_roll = TD_Trainer.Train(yaw_params, pitch_params, roll_params)
    
    end_time = time.time()     
    execution_time = end_time - start_time
    print(f"Training successfully completed!")
    print(f"Time taken: {execution_time} seconds")
    
    np.savez('Factor_Matrices.npz', U_yaw=factors[1], U_pitch=factors[2], U_roll=factors[3], U_id=factors[0])
    np.savez('Trained_data.npz', optimized_yaw=optimized_yaw, optimized_pitch=optimized_pitch, optimized_roll=optimized_roll, CoreTensor=core_tensor, W=W)
    
    
    #=========================================================================================================
    #---------------------------------------- Monitoring block------------------------------------------------
    # ## ploting a column of any factor matrix using optimized cosine params for yaw, pitch and roll rotations
    #=========================================================================================================
    
    # Function definition
    def f(w, a, b, c, d):
        return a * np.cos(b * w + c) + d
    
    if plot_factor_matrices_dims:
    
        # Create the x_sample values from 0 to 360 with intervals of 5 degrees
        x_sample_deg = np.arange(-180, 181, 5)  # Using intervals of 5 degrees
        x_sample_rad = np.radians(x_sample_deg)
        
        optimized_params = [optimized_yaw, optimized_pitch, optimized_roll]
        bins = [yaw_bins, pitch_bins, roll_bins]
        titles = ['yaw','pitch','roll']
        
        for j in range(3):
            # bins_arr= np.array(bins[j])
            # bins_arr_pos = bins_arr[bins_arr>=0]
            # # Convert bins to the range [0, 360]
            # bins_arr_neg = bins_arr[bins_arr<0]+360    
            num_columns = factors[j+1].shape[1]
            for i in range(num_columns):
                
                a, b, c, d = optimized_params[j][i]
                
                # Calculate f(x) for all x_sample values
                y_sample = f(x_sample_rad, a,b,c,d)  # np.radians converts degrees to radians
                    
                # Calculate f(w) for the given parameters for dimension j
                bins_arr= np.array(bins[j])
                # f_values = np.array([f(bins_arr[k], a,b,c,d) for k in range(len(bins_arr))])
            
                # y values of data
                vector = factors[j+1][:, i] # turn first column of yaw matrix to a vector
                # vec_neg = vector[bins_arr<0]
                # vec_pos = vector[bins_arr>=0]
                
                # Increase figure size and resolution
                plt.figure(figsize=(10, 6), dpi=150)
                
                # Plot the function f(x) using optimized params
                plt.plot(x_sample_deg, y_sample, label='f(x) = cos(x)',marker='.', linestyle='-', color='red')
                
                # plot data of this particular dimension
                plt.plot(bins_arr, vector, label='Value for specific angle',
                             marker='o', linestyle='--', color='blue')
                
                # # Plot the values v for the angles k
                # plt.plot(bins_arr_pos, vec_pos, label='positive Values for specific angles',
                #             marker='o', linestyle='--', color='blue')
                
                # # Plot the values v for the angles k
                # plt.plot(bins_arr_neg, vec_neg, label='Negative Values for specific angles',
                #             marker='o', linestyle='--', color='green')
                
                # Labels and title
                plt.xlabel('Angle (degree)')
                plt.ylabel('Function value / v')
                plt.title(f'Plot of {titles[j]} dimension {i} and f(x) = cos(x)')
                
                # Show legend
                plt.legend()
                plt.grid(True)
                # Show the plot
                plt.show()
         
    
    #=====================================
    # ## Constructing the x for first data 
    #=====================================
    # result = np.einsum('ijklm,i,j,k,l->m', W, factors[0][0], factors[1][0], factors[2][0], factors[3][0])
    
    # # Output the result
    # print("Resulting vector shape:", result.shape)
    # print("Resulting vector:", result)
    
    ################################ Step 3: Test #####################################
    
    start_time = time.time()
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)    
    
    
    if test_mode == 1:
        """ 
         Testing single input image to get Euler angles
        """
        
        # existed in training set
        
        singe_test_path = "testSamples/ID1_(30_20_-10).png"
        singe_test_path = "testSamples/frame_00350_rgb.png"    
        val_set_cnt = 1
         
        x = FE.get_feature_vector(face_mesh, singe_test_path, normalized=True) 
        
        u_id_shape = factors[0][1].size
        
        
        # existed in training set
        test_anonotationFile_path = 'D:/datasets/BIWI/db_annotations/10/frame_00479_pose.bin'
        # test_anonotationFile_path = 'D:/datasets/BIWI/db_annotations/10/frame_00486_pose.bin'
        
        # Not existed in training set
        # test_anonotationFile_path = 'E:/datasets/BIWI/db_annotations/12/frame_00362_pose.bin'
        
        true_yaw, true_pitch, true_roll = CreateTensor.read_head_pose_values(test_anonotationFile_path)
        
        # true_yaw, true_pitch, true_roll = np.radians(true_yaw), np.radians(true_pitch), np.radians(true_roll)
        
        print(f'True angles: w_y:{true_yaw} w_p:{true_pitch} w_r:{true_roll}')
        #W[:, :3, :3, :3, :] If after decomposition, the ranks of factor matrices of 1,2, and 3 are huigher than 3, use this code
        #optimized_yaw[0:3,:] only the first 3 dims have noticable 
        est_w_y, est_w_p, est_w_r, est_u_id = TD_Tester.Test(W, x, u_id_shape, optimized_yaw[0:3,:], 
                                                          optimized_pitch[0:3,:], optimized_roll[0:3,:], 
                                                          u_id=factors[0][256], # These extra params are for the sake of debug.
                                                          f_y=factors[1][6],  # If the test sample is within the train set,
                                                          f_p=factors[2][6], # we can provide the corresponding entry of the
                                                          f_r=factors[3][0]) # test in factor matrcies except those that we
                                                                              # want to test by ploting the objective function.
                                                          
        print('Estimated yaw in degree = ', est_w_y)
        print('Estimated pitch in degree = ', est_w_p)
        print('Estimated roll in degree = ', est_w_r) 
                                                     
        end_time = time.time()
        # Plot the captured w_y values and the corresponding objective values
        # plt.plot(w_y_values, objective_values, marker='o')
        # plt.xlabel('w_y')
        # plt.ylabel('Objective function value')
        # plt.title('Objective function value vs w_y')
        # plt.show()
    
    elif test_mode == 2:
        """ 
         Computing MAE for the given validation set
        """
        true_euler_angles = []
        pred_euler_angles = []
        
        val_set_path = "3D_DB_(50_40_30)_valset_mixedExpressions(60_Subjects)" 
        val_set_cnt = 60
        
        u_id_shape = factors[0][1].size
        
        for i in range(val_set_cnt):
            
            if i % 20 == 0:
                print(f'{i} data are valiadted')
                
            curr_folder = str(300 - val_set_cnt + (i+1) )
            
            test_path = os.path.join(val_set_path, curr_folder)  
            
            image_files = [f for f in os.listdir(test_path) if f.endswith('.png')]
            
            # for k in range(5):
            
            # Select a random image
            random_image = random.choice(image_files)
            random_image_path = os.path.join(test_path, random_image)  
             
            x = FE.get_feature_vector(face_mesh, random_image_path, normalized=True) 
            
            # Create the expected pattern to match the part with the numbers, focusing on the ID and parentheses with values
            pattern = rf'ID{curr_folder}_\((-?\d+)_(-?\d+)_(-?\d+)\)'
    
            # Apply the regex match
            match = re.search(pattern, random_image)        
        
            if match:
                # Extract Y, Z, and W as integers
                true_yaw, true_pitch, true_roll = match.groups()
                true_euler_angles.append((float(true_yaw), float(true_pitch), float(true_roll)))
            else:
                print(f'no match at id = {curr_folder}')
                continue              
            
            # true_yaw, true_pitch, true_roll = np.radians(true_yaw), np.radians(true_pitch), np.radians(true_roll)
            
            # print(f'True angles: w_y:{true_yaw} w_p:{true_pitch} w_r:{true_roll}')
            #W[:, :3, :3, :3, :] If after decomposition, the ranks of factor matrices of 1,2, and 3 are huigher than 3, use this code
            #optimized_yaw[0:3,:] only the first 3 dims have noticable 
            est_w_y, est_w_p, est_w_r, est_u_id = TD_Tester.Test(W, x, u_id_shape, optimized_yaw[0:3,:], 
                                                              optimized_pitch[0:3,:], optimized_roll[0:3,:], 
                                                              None, None, None, None) # the params that are None are placeholder for debug use
            pred_euler_angles.append((round(est_w_y, 3) , round(est_w_p,3) , round(est_w_r,3) ))
        
        end_time = time.time()
            
        # Compute the absolute differences for each angle (yaw, pitch, roll)
        yaw_errors = [abs(true[0] - pred[0]) for true, pred in zip(true_euler_angles, pred_euler_angles)]
        pitch_errors = [abs(true[1] - pred[1]) for true, pred in zip(true_euler_angles, pred_euler_angles)]
        roll_errors = [abs(true[2] - pred[2]) for true, pred in zip(true_euler_angles, pred_euler_angles)]
        
        # Compute the Mean Absolute Error (MAE) for each angle
        mae_yaw = np.mean(yaw_errors)
        mae_pitch = np.mean(pitch_errors)
        mae_roll = np.mean(roll_errors)
        
        # Compute overall MAE (you can average across all angles if desired)
        mae_total = (mae_yaw + mae_pitch + mae_roll) / 3
        
        # Output the results
        print(f'MAE (Yaw): {mae_yaw}')
        print(f'MAE (Pitch): {mae_pitch}')
        print(f'MAE (Roll): {mae_roll}')
        print(f'Total MAE: {mae_total}')
            
            
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    elapsed_time /= val_set_cnt
    
    # Convert to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    # Format the elapsed time in hours:minutes:seconds
    elapsed_time_formatted = f"{hours:02}:{minutes:02}:{seconds:02}"
    
    print(f"Average Elapsed time for test: {elapsed_time_formatted}")
    print('Terminated')

if __name__ == "__main__":
    main()


