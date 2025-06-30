# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:55:07 2024

This code is to combine two FFNs. The first FFN is an encoder that gets landmarks and outputs U matrices (U matrices are the output of 
                                                                        tensor decomposition) trained with EncoderTrainer.py
        The second FFN is the one that gets U matrices and predict Euler angles. This FFN is trained using MLPHeadsTrainer.py
        The current script is to combine these two FFNs
        
@author: Mahdi Ghafourian
"""
# Standard library imports
import warnings
import yaml

# Third-party library imports
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# Local application imports
from helpers import FeatureExtractor as FE


class LandmarkEncoder(nn.Module):
    def __init__(self, input_size, matrix_dims):
        super(LandmarkEncoder, self).__init__()
        self.input_size = input_size
        self.matrix_dims = matrix_dims
        self.total_output_size = sum([m * n for m, n in matrix_dims])  # Total size of the latent vector             
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),   
            nn.ReLU(),
            # nn.Dropout(0.05),
        
            nn.Linear(1024, 512),          
            nn.ReLU(),
            # nn.Dropout(0.05),
            
            nn.Linear(512, 256),          
            nn.ReLU(),
            # nn.Dropout(0.05),
            
            nn.Linear(256, 128),          
            nn.ReLU(),
            
            nn.Linear(128, 64),          
            nn.Tanh(),
                                   
            nn.Linear(64, self.total_output_size)
        )                        

    def forward(self, x):
        
        # Encode input landmarks to latent representation
        latent = self.encoder(x)
        
        # Split and reshape into individual matrices
        matrices = []
        start_idx = 0
        for m, n in self.matrix_dims:
            size = m * n
            matrices.append(latent[:, start_idx:start_idx + size].view(-1, m, n))
            start_idx += size
        
        return matrices


class AnglePredictionNetwork(nn.Module):
    def __init__(self, input_size):
        super(AnglePredictionNetwork, self).__init__()
        self.input_size = input_size                     
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.05),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1),  # Output a single value
        )
        
        # Initialize weights using Xavier initialization
        self.apply(self.init_weights)
        
    def init_weights(self, m):
       # Apply Xavier initialization to linear layers
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

class CombinedAnglePredictionModel(nn.Module):
    def __init__(self, encoder, yaw_network, pitch_network, roll_network):
        super(CombinedAnglePredictionModel, self).__init__()
        self.encoder = encoder
        self.yaw_network = yaw_network
        self.pitch_network = pitch_network
        self.roll_network = roll_network

    def forward(self,input_landmarks):#, yaw_input, pitch_input, roll_input):
        predicted_matrices = self.encoder(input_landmarks)

        yaw_input = predicted_matrices[0].squeeze(1)
        pitch_input = predicted_matrices[1].squeeze(1)
        roll_input = predicted_matrices[2].squeeze(1)
                
        yaw_output = self.yaw_network(yaw_input)  # Predict yaw
        pitch_output = self.pitch_network(pitch_input)  # Predict pitch
        roll_output = self.roll_network(roll_input)  # Predict roll
        # return torch.rad2deg(yaw_output), torch.rad2deg(pitch_output), torch.rad2deg(roll_output)
        return yaw_output, pitch_output, roll_output
    
#==========================================================================================


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


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) 
    
def model_builder():
    
    #==========================================================================================
    warnings.filterwarnings("ignore")
    
    # Load config
    config = load_config("configs/config_EncoderTrainer.yaml")
    
    input_size = config["input_size"]
    
    loaded_data = np.load('outputs/features/Trained_data.npz')
    optimized_yaw = loaded_data['optimized_yaw']
    optimized_pitch = loaded_data['optimized_pitch']
    optimized_roll = loaded_data['optimized_roll']
    
    yaw_inputSize = optimized_yaw.shape[0]
    pitch_inputSize = optimized_pitch.shape[0]
    roll_inputSize = optimized_roll.shape[0]
    
    loaded_data = np.load('outputs/features/Factor_Matrices.npz')
    U_yaw = loaded_data['U_yaw']
    U_pitch = loaded_data['U_pitch']
    U_roll = loaded_data['U_roll']
    
    matrix_dims = [ (1, U_yaw.shape[1]),
                    (1, U_pitch.shape[1]),
                    (1, U_roll.shape[1])
                    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #--------------------------------- Load encoder network ------------------------------------
    # Instantiate autoencoder
    encoder = LandmarkEncoder(input_size, matrix_dims).to(device)
    encoder.load_state_dict(torch.load('models/Encoder.pth'))
    
    # encoder.eval()
    
    #-------------------------------------------------------------------------------------------
    #------------------------------ Load predictor networks ------------------------------------
    # Instantiate individual networks
    yaw_network = AnglePredictionNetwork(yaw_inputSize)
    pitch_network = AnglePredictionNetwork(pitch_inputSize)
    roll_network = AnglePredictionNetwork(roll_inputSize)
    
    # Load pre-trained weights
    yaw_network.load_state_dict(torch.load('models/yaw_network.pth'))
    pitch_network.load_state_dict(torch.load('models/pitch_network.pth'))
    roll_network.load_state_dict(torch.load('models/roll_network.pth'))
    
    # Create the combined model
    combined_model = CombinedAnglePredictionModel(encoder, yaw_network, pitch_network, roll_network).to(device)

    combined_model.to(device)
    scripted_model = torch.jit.script(combined_model)  # or torch.jit.trace() if model is not dynamic
    scripted_model.save("models/combined_model_scripted.pth")
    print("model is built")

if __name__ == "__main__":   
    
    model_builder()

