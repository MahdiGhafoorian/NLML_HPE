# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:35:32 2024

This is the script to train an Ensemble of three networks to learn the mapping from cosine function vector to the corresponding angle

@author: Mahdi Ghafourian
"""

# Standard library
import warnings
import yaml

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the architecture of the network
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
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.05),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.05),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.05),
            
            nn.Linear(64, 1),  # Output a single value
        )
        
        # Initialize weights using Xavier initialization
        self.apply(self.init_weights)
        
    def init_weights(self, m):
       # Apply Xavier initialization to linear layers
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.uniform_(m.weight, a=1e-5, b=1e-3)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
    
def cosine(w_i, a_j, b_j, c_j, d_j):
    # radian = (w_i * 3.14) / 180
    return a_j * np.cos(b_j * w_i + c_j) + d_j

# Function to split data into train and validation sets
def split_data(U_matrix, angle_vector, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        U_matrix, angle_vector, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val

# Training function
def train_network(network, train_loader, val_loader, epochs, criterion, optimizer, device, networkName):
    best_val_loss = float('inf')  
    best_model_state = None
    best_epoch = None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    network.to(device)
    for epoch in range(epochs):
        # Training
        network.train()
        train_loss = 0.0
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = network(batch_X)
            # print(outputs[:5])
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()            
            
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = network(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        # Save the model if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = network.state_dict()  # Save the model's state_dict
            best_epoch = epoch + 1
            
        scheduler.step()
        
    # After training, save the best model
    torch.save(best_model_state, 'models/'+networkName+'_network.pth')
    print(f'Best Model Saved for {networkName} at Epoch {best_epoch}\n')
    
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)   
    
def train_MlpHeads():
    
    warnings.filterwarnings("ignore")
    
    # Load config
    config = load_config("configs/config_MlpHeads.yaml")
    
    yaw_min_bin = config["yaw_bins"]["min_bin"]
    yaw_max_bin = config["yaw_bins"]["max_bin"]
    yaw_interval = config["yaw_bins"]["interval"]
    
    pitch_min_bin = config["pitch_bins"]["min_bin"]
    pitch_max_bin = config["pitch_bins"]["max_bin"]
    pitch_interval = config["pitch_bins"]["interval"]
    
    roll_min_bin = config["roll_bins"]["min_bin"]
    roll_max_bin = config["roll_bins"]["max_bin"]
    roll_interval = config["roll_bins"]["interval"]
    
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = float(config["lr"])
     
    # Example data: Replace with your actual U_yaw, U_pitch, U_roll, and ground truth angles
    # n_samples = 1000
    # U_yaw = np.random.rand(n_samples, 3)
    # U_pitch = np.random.rand(n_samples, 3)
    # U_roll = np.random.rand(n_samples, 3)
    # yaw_angles = np.random.rand(n_samples, 1)
    # pitch_angles = np.random.rand(n_samples, 1)
    # roll_angles = np.random.rand(n_samples, 1)
    
    loaded_data = np.load('outputs/features/Trained_data.npz')
    optimized_yaw = loaded_data['optimized_yaw']
    optimized_pitch = loaded_data['optimized_pitch']
    optimized_roll = loaded_data['optimized_roll']
    
    # Get the number of dimension of each euler angle for which we train
    yaw_inputSize = optimized_yaw.shape[0]
    pitch_inputSize = optimized_pitch.shape[0]
    roll_inputSize = optimized_roll.shape[0]
    
    # Generate grount truth angles in radian within desired range 
    # gt_yaw_angles = np.radians(np.arange(-50, 50.001, 0.001).astype(np.float32))
    # gt_pitch_angles = np.radians(np.arange(-40, 40.001, 0.001).astype(np.float32))
    # gt_roll_angles = np.radians(np.arange(-30, 30.001, 0.001).astype(np.float32))
    
    gt_yaw_angles = np.radians(np.arange(yaw_min_bin, yaw_max_bin, yaw_interval).astype(np.float32))
    gt_pitch_angles = np.radians(np.arange(pitch_min_bin, pitch_max_bin, pitch_interval).astype(np.float32))
    gt_roll_angles = np.radians(np.arange(roll_min_bin, roll_max_bin, roll_interval).astype(np.float32))
    
    U_yaw = np.zeros((len(gt_yaw_angles),yaw_inputSize))
    U_pitch = np.zeros((len(gt_pitch_angles),pitch_inputSize))
    U_roll = np.zeros((len(gt_roll_angles),roll_inputSize))
    
    #========================================================================================================
    # Create the training set by filling the following matrices for the given angles and optimized params
    for i, w in enumerate(gt_yaw_angles):
        # w_rd = np.radians(w)
        for j, row in enumerate(optimized_yaw):
            a,b,c,d = row
            U_yaw[i][j] = cosine(w, a, b, c, d)
            
    for i, w in enumerate(gt_pitch_angles):
        # w_rd = np.radians(w)
        for j, row in enumerate(optimized_pitch):
            a,b,c,d = row
            U_pitch[i][j] = cosine(w, a, b, c, d)
            
    for i, w in enumerate(gt_roll_angles):
        # w_rd = np.radians(w)
        for j, row in enumerate(optimized_roll):
            a,b,c,d = row
            U_roll[i][j] = cosine(w, a, b, c, d)            
    #========================================================================================================
        
    # Split datasets
    X_yaw_train, X_yaw_val, y_yaw_train, y_yaw_val = split_data(U_yaw, gt_yaw_angles)
    X_pitch_train, X_pitch_val, y_pitch_train, y_pitch_val = split_data(U_pitch, gt_pitch_angles)
    X_roll_train, X_roll_val, y_roll_train, y_roll_val = split_data(U_roll, gt_roll_angles)
    
    # Hyperparameters
    # epochs = 50
    # batch_size = 256
    # lr = 5e-3
    
    # Convert data to PyTorch tensors and create DataLoaders
    def create_dataloader(X_train, y_train, X_val, y_val):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    # create dataloaders
    yaw_train_loader, yaw_val_loader = create_dataloader(X_yaw_train, y_yaw_train, X_yaw_val, y_yaw_val)
    pitch_train_loader, pitch_val_loader = create_dataloader(X_pitch_train, y_pitch_train, X_pitch_val, y_pitch_val)
    roll_train_loader, roll_val_loader = create_dataloader(X_roll_train, y_roll_train, X_roll_val, y_roll_val)
    
    # Instantiate networks, loss function, and optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    yaw_network = AnglePredictionNetwork(yaw_inputSize)
    pitch_network = AnglePredictionNetwork(pitch_inputSize)
    roll_network = AnglePredictionNetwork(roll_inputSize)
    
    criterion = nn.MSELoss()
    yaw_optimizer = optim.Adam(yaw_network.parameters(), lr=lr, weight_decay=1e-5) # momentum=0.9)
    pitch_optimizer = optim.Adam(pitch_network.parameters(), lr=lr, weight_decay=1e-5)
    roll_optimizer = optim.Adam(roll_network.parameters(), lr=lr, weight_decay=1e-5)
    
    # Train the networks
    print("Training yaw network...\n")
    train_network(yaw_network, yaw_train_loader, yaw_val_loader, epochs, criterion, yaw_optimizer, device, 'yaw')
    
    print("Training pitch network...")
    train_network(pitch_network, pitch_train_loader, pitch_val_loader, epochs, criterion, pitch_optimizer, device, 'pitch')
    
    print("Training roll network...")
    train_network(roll_network, roll_train_loader, roll_val_loader, epochs, criterion, roll_optimizer, device, 'roll')

if __name__ == "__main__":   
    
    train_MlpHeads()
