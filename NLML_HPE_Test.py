# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:16:19 2025

@author: Mahdi Ghafourian
"""

# Standard Library Imports
import os
import time
import math
import re
import warnings

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import torch
import yaml
import mediapipe as mp

# Helper Modules Imports
from helpers import datasets
from helpers import FeatureExtractor as FE


def W300_EulerAngles2Vectors(rx, ry, rz):
        '''
        rx: pitch
        ry: yaw
        rz: roll
        '''
        
        # Convert to radians
        rx = np.radians(rx)
        ry = np.radians(ry)
        rz = np.radians(rz)
        
        ry *= -1
        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(rx), -np.sin(rx)],
                        [0.0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                        [np.sin(rz), np.cos(rz), 0.0],
                        [0.0, 0.0, 1.0]])
                        
        R = R_x @ R_y @ R_z
        
        l_vec = R @ np.array([1, 0, 0]).T
        b_vec = R @ np.array([0, 1, 0]).T
        f_vec = R @ np.array([0, 0, 1]).T
        return R, l_vec, b_vec, f_vec
 
    
# Function to compute the MAEV (Mean Absolute Error between columns)
def compute_maev(ground_truth, predicted):
    v1_err_sum = 0
    v2_err_sum = 0
    v3_err_sum = 0
    count = len(ground_truth)
    c_rad2deg = 180.0 / np.pi
    
    for i in range(count):
        # Extract yaw, pitch, roll from ground truth and predicted
        yaw_gt, pitch_gt, roll_gt = ground_truth[i]
        yaw_pred, pitch_pred, roll_pred = predicted[i]
        
        # Convert both to rotation matrices
        R_gt, l_gt_vec, b_gt_vec, f_gt_vec = W300_EulerAngles2Vectors( pitch_gt, yaw_gt, roll_gt)
        R_pred, l_pred_vec, b_pred_vec, f_pred_vec = W300_EulerAngles2Vectors(pitch_pred, yaw_pred, roll_pred)
        
        # Vector errors
        v1_err = math.acos(np.clip(np.sum(l_gt_vec * l_pred_vec), -1, 1)) * c_rad2deg
        v2_err = math.acos(np.clip(np.sum(b_gt_vec * b_pred_vec), -1, 1)) * c_rad2deg
        v3_err = math.acos(np.clip(np.sum(f_gt_vec * f_pred_vec), -1, 1)) * c_rad2deg
        
        v1_err_sum += v1_err
        v2_err_sum += v2_err
        v3_err_sum += v3_err        
            
    # Compute MAEV: Mean of the absolute errors of all samples
    MAEV = (v1_err_sum + v2_err_sum + v3_err_sum) / (3 * count)
    v_left_error = v1_err_sum / count
    v_down_error = v2_err_sum / count
    v_front_error = v3_err_sum / count
    
    return MAEV, v_left_error, v_down_error, v_front_error
    
def compute_errors(true_angles, pred_angles):

    # Compute the absolute differences for each angle (yaw, pitch, roll)
    yaw_errors = [abs(true[0] - pred[0]) for true, pred in zip(true_angles, pred_angles)]
    pitch_errors = [abs(true[1] - pred[1]) for true, pred in zip(true_angles, pred_angles)]
    roll_errors = [abs(true[2] - pred[2]) for true, pred in zip(true_angles, pred_angles)]
    
    # Compute the Mean Absolute Error (MAE) for each angle
    mae_yaw = np.mean(yaw_errors)
    mae_pitch = np.mean(pitch_errors)
    mae_roll = np.mean(roll_errors)
    
    # Compute the Standard Deviation of the absolute errors for each component
    
    std_abs_error_yaw = np.std(yaw_errors, ddof=1)
    std_abs_error_pitch = np.std(pitch_errors, ddof=1)
    std_abs_error_roll = np.std(roll_errors, ddof=1)
    
    # Compute overall MAE (you can average across all angles if desired)  maev = np.mean(np.linalg.norm(preds - gt, axis=1))
    mae_total = (mae_yaw + mae_pitch + mae_roll) / 3
    
    
    maev, v_left_error, v_down_error, v_front_error = compute_maev(true_angles, pred_angles)
    
    # Output the results
    print(f'MAE (Yaw): {mae_yaw:.2f}')
    print(f'MAE (Pitch): {mae_pitch:.2f}')
    print(f'MAE (Roll): {mae_roll:.2f}')
    print(f'Total MAE: {mae_total:.2f}')
    print(f'MAEV: {maev:.2f}')
    print(f'Left vector Error (red): {v_left_error:.2f}')
    print(f'Down vector Error (green): {v_down_error:.2f}')
    print(f'Front vector Error (blue): {v_front_error:.2f}')
    print(f'std (Yaw): {std_abs_error_yaw:.2f}')
    print(f'std (Pitch): {std_abs_error_pitch:.2f}')
    print(f'std (Roll): {std_abs_error_roll:.2f}')    
    
    
def compute_interval_mae(true_angles, pred_angles_list, labels,  yaw_intervals, pitch_intervals, roll_intervals):
    true_angles = np.array(true_angles)
        
    results = {}
    
    markers = ['o', 'v', 'x']  # Different markers for each line
    colors = ['b', 'g', 'r']  # Different colors for each line
    
    def compute_mae_for_intervals(name, index, intervals):
        plt.figure()
        for pred_angles, label, marker, color in zip(pred_angles_list, labels, markers, colors):
            pred_angles = np.array(pred_angles)
            maes = []
            centers = []
            for low, high in intervals:
                mask = (true_angles[:, index] >= low) & (true_angles[:, index] < high)
                if np.any(mask):  # Ensure there are samples in this range
                    mae = mean_absolute_error(true_angles[mask, index], pred_angles[mask, index])
                    results[f"{name} ({low}, {high}) - {label}"] = mae
                    maes.append(mae)
                    centers.append((low + high) / 2)
            
            if maes:
                plt.plot(centers, maes, marker=marker, linestyle='-', color=color, label=label)
        
        plt.xlabel(f"Ground truth {name} (deg)")
        plt.ylabel("MAE (deg)")
        # plt.title(f"MAE for {name} Intervals")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    compute_mae_for_intervals("Yaw", 0, yaw_intervals)
    compute_mae_for_intervals("Pitch", 1, pitch_intervals)
    compute_mae_for_intervals("Roll", 2, roll_intervals)
    
    return results
    
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) 

## main *****************************************************************************

# show histogram
# error bins intervals
# val_set_path
# method

def NLML_HPE_Tester():
    
    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config
    config = load_config("configs/config_EncoderTrainer.yaml")
    
    yaw_min_bin = config["yaw_bins"]["min_bin"]
    yaw_max_bin = config["yaw_bins"]["max_bin"]
    
    pitch_min_bin = config["pitch_bins"]["min_bin"]
    pitch_max_bin = config["pitch_bins"]["max_bin"]
    
    roll_min_bin = config["roll_bins"]["min_bin"]
    roll_max_bin = config["roll_bins"]["max_bin"]
    
    
    config = load_config("configs/config_NLML_HPE_Test.yaml")    
    
    yaw_intervals = [tuple(x) for x in config["yaw_intervals"]]
    pitch_intervals = [tuple(x) for x in config["pitch_intervals"]]
    roll_intervals = [tuple(x) for x in config["roll_intervals"]]
    
    val_set = config["val_set"]
    val_set_path = config["val_set_path"]   
    
    
    ###############################################################################
    ###############################################################################
    ############## Load our model #################################################
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    
    NLML_HPE_model = torch.jit.load("models/combined_model_scripted.pth", map_location=device)
    NLML_HPE_model = NLML_HPE_model.to(device)
    NLML_HPE_model.eval()
    
    
    ###############################################################################
    ###############################################################################
    
    
    true_angles = []
    pred_angles_NLML_HPE = []
    
    # val_set_path = "3D_DB_(50_40_30)_valset_singleExpressions_180_Subjects" 
    # # val_set_path = 'D:/datasets/BIWI/faces_0/'
    # # val_set_path= 'C:\\Mahdi\\development\\TokenHPE-main\\datasets\\AFLW2000'
    
    
    # val_set = 'facescape' # facescape / biwi / AFLW2000
    
    not_detected_lndmrks = 0
    outside_range_smpls = 0
    processed_smpls = 0
    
    start_time = time.time()
       
    
    if val_set == 'biwi':
    ######## BIWI val set ##############
        # data = np.load('C:\\Mahdi\\development\\TokenHPE-main\\datasets\\BIWI.npz')
        data = np.load(val_set_path)
        images = data['image']  # Array of cropped face images
        poses = data['pose']        
             
        for i in range(len(images)):
            true_yaw, true_pitch, true_roll = poses[i]
            
            if not (yaw_min_bin <= true_yaw <= yaw_max_bin and pitch_min_bin <= true_pitch <= pitch_max_bin and roll_min_bin <= true_roll <= roll_max_bin):
                continue
            
            test_landmarks = FE.get_feature_vector_from_image(face_mesh, images[i], normalize=True, isPil=False) 
            
            all_zero = (test_landmarks == 0).all()
            if(all_zero.item()): # if landmarks isn't extracted
                not_detected_lndmrks += 1
                continue
            
            test_landmarks = test_landmarks.unsqueeze(0).float().to(device)       
            
            processed_smpls += 1
            
            true_angles.append((float(true_yaw), float(true_pitch), float(true_roll)))
            ###############################################################################
            try:
                
                # Perform prediction
                with torch.no_grad():
                    predicted_yaw, predicted_pitch, predicted_roll = NLML_HPE_model(test_landmarks)                   
                    pred_angles_NLML_HPE.append((round(np.degrees(predicted_yaw.item()), 3) , round(np.degrees(predicted_pitch.item()),3) , round(np.degrees(predicted_roll.item()),3) ))  
                                       
            except Exception as e:
                # Handle any other exceptions
                print(f"An unexpected error occurred: {e}")
           
            ################################################
                    
            
    ######## BIWI val set ############## 
    
    elif val_set == 'facescape': # not use biwi    
    
    ######## Facescape val set ##############
                     
        cnt = 0
        ###################################################
                
        folders = [name for name in os.listdir(val_set_path) if os.path.isdir(os.path.join(val_set_path, name))]
         
        for curr_folder in folders:
            test_path = os.path.join(val_set_path, curr_folder)  
            image_files = [f for f in os.listdir(test_path) if f.endswith('.png') or f.endswith('.jpg')]
       
            for image in image_files:
                cnt += 1
                
                if cnt % 100 == 0:
                    print(f'{cnt} data are valiadted')
                                    
                image_path = os.path.join(test_path, image) 
                test_landmarks = FE.get_feature_vector(face_mesh, image_path, normalize=True)             
                            
                test_landmarks = test_landmarks.unsqueeze(0).float().to(device)
                
                processed_smpls += 1
                # Create the expected pattern to match the part with the numbers, focusing on the ID and parentheses with values
                pattern = rf'ID{curr_folder}_\((-?\d+)_(-?\d+)_(-?\d+)\)'
            
                # Apply the regex match
                match = re.search(pattern, image_path)        
            
                if match:
                    # Extract Y, Z, and W as integers
                    true_yaw, true_pitch, true_roll = match.groups()
                    true_angles.append((float(true_yaw), float(true_pitch), float(true_roll)))                
                else:
                    print(f'no match at id = {curr_folder}')
                    continue
                
                try:
                
                    # Perform prediction
                    with torch.no_grad():
                        predicted_yaw, predicted_pitch, predicted_roll = NLML_HPE_model(test_landmarks)                
                        pred_angles_NLML_HPE.append((round(np.degrees(predicted_yaw.item()), 3) , round(np.degrees(predicted_pitch.item()),3) , round(np.degrees(predicted_roll.item()),3) ))  
                                            
                except Exception as e:
                    # Handle any other exceptions
                    print(f"An unexpected error occurred: {e}")            
        
             
        # Function to count values in each interval
        def count_in_intervals(values, intervals):
            counts = []
            for low, high in intervals:
                counts.append(sum(low <= v < high for v in values))  # List comprehension for lists
            return counts
        
        # Extract yaw, pitch, and roll from list
        yaw_values, pitch_values, roll_values = zip(*true_angles)  # Unpack into separate lists
        
        # Count occurrences in each interval
        yaw_counts = count_in_intervals(yaw_values, yaw_intervals)
        pitch_counts = count_in_intervals(pitch_values, pitch_intervals)
        roll_counts = count_in_intervals(roll_values, roll_intervals)
        
        # Plot function
        def plot_Histogram(intervals, counts, title, xlabel):
            labels = [f"{low:.2f} to {high:.2f}" for low, high in intervals]
            plt.figure(figsize=(8, 5))
            plt.bar(labels, counts, color='skyblue', edgecolor='black')
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(xlabel)
            plt.ylabel("Count")
            plt.title(title)
            plt.grid(axis='y', linestyle="--", alpha=0.7)
            plt.show()
        
        # Plot distributions
        plot_Histogram(yaw_intervals, yaw_counts, "Yaw Histogram", "Yaw Intervals")
        plot_Histogram(pitch_intervals, pitch_counts, "Pitch Histogram", "Pitch Intervals")
        plot_Histogram(roll_intervals, roll_counts, "Roll Histogram", "Roll Intervals")
        print('cnt=',cnt)
                
                ################################################
                
                
    ######## Facescape val set ##############
    elif val_set == 'AFLW2000':
    ######## AFLW2000 val set ##############    
        
        filename_list = val_set_path + '\\files.txt'
        pose_dataset = datasets.getDataset(val_set, val_set_path, filename_list, transformations=None, train_mode=False)
    
        val_loader = torch.utils.data.DataLoader(
            dataset=pose_dataset,
            batch_size=1,
            num_workers=0)
        
        
        for i, (images, r_label, cont_labels, name) in enumerate(val_loader):
            
            # gt euler
            true_yaw = cont_labels[:, 0].float() * 180 / np.pi # yaw
            true_pitch = cont_labels[:, 1].float() * 180 / np.pi # pitch
            true_roll = cont_labels[:, 2].float() * 180 / np.pi # roll
            
            if not (yaw_min_bin <= true_yaw <= yaw_max_bin and pitch_min_bin <= true_pitch <= pitch_max_bin and roll_min_bin <= true_roll <= roll_max_bin):
                continue        
            
            test_landmarks = FE.get_feature_vector_from_image(face_mesh, images.squeeze(), normalize=True, isPil=True) 
            
            all_zero = (test_landmarks == 0).all()
            if(all_zero.item()): # if landmarks isn't extracted
                not_detected_lndmrks += 1
                continue
            
            test_landmarks = test_landmarks.unsqueeze(0).float().to(device)                    
            
            processed_smpls += 1
                        
            true_angles.append((float(true_yaw), float(true_pitch), float(true_roll)))
            ###############################################################################
            try:
                
                # Perform prediction
                with torch.no_grad():
                    predicted_yaw, predicted_pitch, predicted_roll = NLML_HPE_model(test_landmarks)                    
                    pred_angles_NLML_HPE.append((round(np.degrees(predicted_yaw.item()), 3) , round(np.degrees(predicted_pitch.item()),3) , round(np.degrees(predicted_roll.item()),3) ))  
                       
                    
            except Exception as e:
                # Handle any other exceptions
                print(f"An unexpected error occurred: {e}")
           
            ################################################
            
            
        ######## AFLW2000 val set ##############
            
            
    end_time = time.time()    
    
    # Compute metrics for each set of predicted angles
    print("=============================Metrics for pred_angles_NLML_HPE:")
    compute_errors(true_angles, pred_angles_NLML_HPE)
    
    print("\n======================================================================")
    
    pred_angles_list = [pred_angles_NLML_HPE]   # this is the original code and must be uncommented
    labels = ["NLML_HPE"]
    mae_results = compute_interval_mae(true_angles, pred_angles_list, labels, yaw_intervals, pitch_intervals, roll_intervals) 
    # print(mae_results)
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    # Convert to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    # Format the elapsed time in hours:minutes:seconds
    elapsed_time_formatted = f"{hours:02}:{minutes:02}:{seconds:02}"
    
    print(f"Average Elapsed time for test: {elapsed_time_formatted}")


if __name__ == "__main__":   
    
    NLML_HPE_Tester()
    