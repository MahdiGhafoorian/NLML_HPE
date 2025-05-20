# -*- coding: utf-8 -*-
"""
Created on Fri May 16 14:32:10 2025

@author: Mahdi Ghafourian
"""

# Basic required packages
import time
import numpy as np
import mediapipe as mp
import argparse
import warnings

# our modules
import TD_Tester
from helpers import FeatureExtractor as FE


def inference():    
       
    """ 
     Testing single input image to get Euler angles
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    parser = argparse.ArgumentParser(description="Inference the head pose of single input image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)    
    
   
    # singe_test_path = "testSamples/ID1_(30_20_-10).png"
     
    x = FE.get_feature_vector(face_mesh, args.image_path, normalize=True) 
    
    
    trained_data = np.load('./outputs/features/Trained_data.npz')

    # Retrieve the arrays
    optimized_yaw = trained_data['optimized_yaw']
    optimized_pitch = trained_data['optimized_pitch']
    optimized_roll = trained_data['optimized_roll']
    core_tensor = trained_data['CoreTensor']
    W = trained_data['W']
    
    factors_data = np.load('./outputs/features/Factor_Matrices.npz')
    
    u_id_shape = factors_data['U_id'][1].size        
    

    #W[:, :3, :3, :3, :] If after decomposition, the ranks of factor matrices of 1,2, and 3 are huigher than 3, use this code
    #optimized_yaw[0:3,:] only the first 3 dims have noticable 
    est_w_y, est_w_p, est_w_r, est_u_id = TD_Tester.Test(W, x, u_id_shape, optimized_yaw[0:3,:], 
                                                      optimized_pitch[0:3,:], optimized_roll[0:3,:], 
                                                      None, None, None, None)
                                                      # u_id=factors[0][256], # These extra params are for the sake of debug.
                                                      # f_y=factors[1][6],  # If the test sample is within the train set,
                                                      # f_p=factors[2][6], # we can provide the corresponding entry of the
                                                      # f_r=factors[3][0]) # test in factor matrcies except those that we
                                                                          # want to test by ploting the objective function.
                                           
    print(f'Estimated yaw in degree = {est_w_y:.2f}')
    print(f'Estimated pitch in degree = {est_w_p:.2f}')
    print(f'Estimated roll in degree = {est_w_r:.2f}') 
                                                 
    end_time = time.time()
    # Plot the captured w_y values and the corresponding objective values
    # plt.plot(w_y_values, objective_values, marker='o')
    # plt.xlabel('w_y')
    # plt.ylabel('Objective function value')
    # plt.title('Objective function value vs w_y')
    # plt.show()    
    
        
if __name__ == "__main__":   
    
    inference()
