# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:56:48 2024

Code to to train a Neural Network for learning angles given the previously trained parameters of Trignometric functions using an 
 encoder

@author: Usuario
"""

# Standard Library
import os
import warnings

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import yaml
import re

# PyTorch and Related Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# Local Imports
from helpers import FeatureExtractor as FE


warnings.filterwarnings("ignore")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # Initialize weights using Xavier initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
       # Apply Xavier initialization to linear layers
        if isinstance(m, nn.Linear):
              # nn.init.xavier_uniform_(m.weight)  # Use Xavier uniform initialization
              nn.init.uniform_(m.weight, a=1e-5, b=1e-3)
              if m.bias is not None:
                  nn.init.zeros_(m.bias)  # Initialize biases to zero
        
    def forward(self, x):        
        latent = self.encoder(x)
        
        # Split and reshape into individual matrices
        matrices = []
        start_idx = 0
        for m, n in self.matrix_dims:
            size = m * n
            matrices.append(latent[:, start_idx:start_idx + size].view(-1, m, n))
            start_idx += size
        
        return matrices

# Dataset class for loading images, extracting landmarks, and providing labels
class LandmarkDataset(Dataset):
    def __init__(self, data=None, labels=None, data_path=None, normalize=True, yaw_bins=None, pitch_bins=None, roll_bins=None):
        self.normalize = normalize
        
        # Set default bins if none provided
        self.yaw_bins = yaw_bins if yaw_bins is not None else np.arange(-50, 51, 10).astype(np.float32)
        self.pitch_bins = pitch_bins if pitch_bins is not None else np.arange(-40, 41, 10).astype(np.float32)
        self.roll_bins = roll_bins if roll_bins is not None else np.arange(-30, 31, 10).astype(np.float32)
        
        if data is not None and labels is not None:
            # Data was preloaded
            self.data = data
            self.labels = labels
        else:
            self.data = []
            self.labels = []
            self.data_path = data_path
            self._load_data()
            
        # if self.normalize:
        #     self._normalize_data()
            
    def _load_data(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)    
        
        identities = os.listdir(self.data_path)
        identities.sort(key=int)
        
        for U_id, folder in enumerate(identities):
            print(f'Landmarks of subject {U_id} with name {folder} are read')
            folder_path = os.path.join(self.data_path, folder)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                try:
                    # Remove "ID(num1)_" prefix and ".png" suffix, then split to get pose values
                    # main_part = image_name.replace('.png', '').split('_(')[1]  # Extract "(num2)_(num3)_(num4)"
                    # Remove outer parentheses and split to get wy, wp, wr
                    # wy, wp, wr = map(float, main_part.strip('()').split('_'))
                    wy, wp, wr = extract_angles(image_name)
                    
                    wy_idx = np.where(self.yaw_bins == wy)[0][0]
                    wp_idx = np.where(self.pitch_bins == wp)[0][0]
                    wr_idx = np.where(self.roll_bins == wr)[0][0]
                    
                    # Extract landmarks
                    # landmarks = self.extract_landmarks(image_path)
                    landmarks = FE.get_feature_vector(face_mesh, image_path, normalize=True)
                    if landmarks is not None:
                        self.data.append((landmarks.float(), torch.tensor([wy_idx, wp_idx, wr_idx, U_id])))
                except (ValueError, IndexError):
                    print(f"Could not parse pose values from image name: {image_name}")

    def _normalize_data(self):
                
        # Compute the global min and max across each feature (dimension) across all landmarks
        min_vals = torch.min(self.data, dim=0)[0]
        max_vals = torch.max(self.data, dim=0)[0]
        
        # Avoid division by zero for columns with constant values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
                
        # Normalize to (0, 1), then scale to (-1, 1)
        self.data = 2 * ((self.data - min_vals) / range_vals) - 1
        
        torch.save(min_vals, 'min_vals.pth')
        torch.save(max_vals, 'max_vals.pth')
        pass
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       # Check if data is preloaded or newly loaded from images
       if isinstance(self.data, list):  # Newly loaded
           landmarks, poses = self.data[idx]
       else:  # Preloaded tensor data
           landmarks = self.data[idx]
           poses = self.labels[idx]
           
       return landmarks.to(device), poses.to(device)

def save_dataset(dataset, file_path):
    # Save each tuple (landmarks, poses) without additional nesting
    data, labels = zip(*dataset)  # Unzip into separate tuples
    data = torch.stack(data)      # Stack landmarks
    labels = torch.stack(labels)  # Stack poses
    torch.save((data, labels), file_path)

# Load dataset function
def load_dataset(file_path):
    data, labels = torch.load(file_path)
    # Return as a LandmarkDataset object with pre-loaded data and labels
    return LandmarkDataset(data=data, labels=labels)

# Define metrics computation
def compute_metrics(predictions, ground_truths):
    """Compute MAE, MSE, and RMSE between predictions and ground truths."""
    mae = torch.mean(torch.abs(predictions - ground_truths)).item()
    mse = torch.mean((predictions - ground_truths) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    return mae, mse, rmse

# helper function for debuging
def analyze_activations(model, input_data, threshold=1e-6, bins=50):
    """
    Analyze activations of a model to detect dead neurons and plot histograms.

    Args:
        model (nn.Module): The neural network model (e.g., LandmarkEncoder).
        input_data (torch.Tensor): Input tensor to test activations.
        threshold (float): Threshold to consider a neuron as "dead" (default is 1e-6).
        bins (int): Number of bins for the histogram (default is 50).

    Returns:
        dead_neurons: Dictionary mapping layer names to the number of dead neurons.
    """
    activations = {}  # Store activations for each layer
    dead_neurons = {}  # Store dead neuron counts per layer
    hooks = []  # Store hooks to remove them later

    def activation_hook(name):
        def hook(module, input, output):
            # Save activations and detect dead neurons
            output_np = output.detach().cpu().numpy()
            activations[name] = output_np
            total_neurons = output.size(1)
            dead_count = (torch.abs(output) < threshold).sum(dim=0).tolist()
            dead_neurons[name] = dead_count.count(total_neurons)  # Count fully dead neurons
        return hook

    # Register hooks for layers in the encoder
    for name, layer in model.encoder.named_modules():
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(activation_hook(f"encoder_{name}")))

    # Register hooks for layers in the decoder
    for name, layer in model.decoder.named_modules():
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(activation_hook(f"decoder_{name}")))

    # Forward pass to trigger hooks
    with torch.no_grad():
        model(input_data)

    # Remove all hooks
    for hook in hooks:
        hook.remove()

    # Plot histograms for each layer
    fig, axes = plt.subplots(len(activations), 1, figsize=(10, 20))
    for i, (layer_name, activation) in enumerate(activations.items()):
        ax = axes[i]
        ax.hist(activation.flatten(), bins=bins, alpha=0.75, color='blue')
        ax.set_title(f"Activation Histogram: {layer_name}")
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    return dead_neurons
    

def generalized_cosine_penalty(outputs, inputs, amplitudes, frequencies, phases, vertical_shifts):
    """
    Compute the cosine penalty for outputs of the model based on a generalized cosine function.

    Args:
        outputs (torch.Tensor): Model predictions, shape (batch_size, 3)
        inputs (torch.Tensor): Input features, shape (batch_size, input_size)
        amplitudes (torch.Tensor): Amplitudes for the cosine functions, shape (3,)
        frequencies (torch.Tensor): Frequencies for the cosine functions, shape (3,)
        phases (torch.Tensor): Phase shifts for the cosine functions, shape (3,)
        vertical_shifts (torch.Tensor): Vertical shifts for the cosine functions, shape (3,)

    Returns:
        torch.Tensor: The cosine penalty value
    """
    # Compute the "x" values for the cosine functions (e.g., mean of inputs or a specific feature)
    x_values = inputs.mean(dim=1).unsqueeze(1)  # Example: Use the mean of input features as x

    # Compute the expected generalized cosine values
    expected_cosines = amplitudes * torch.cos(frequencies * x_values + phases) + vertical_shifts

    # Compute the penalty as the mean squared error between outputs and expected cosines
    penalty = torch.mean((outputs - expected_cosines) ** 2)

    return penalty

def extract_angles(image_name):
    match = re.search(r'_\( *(-?\d+)_(-?\d+)_(-?\d+)\)', image_name)
    if match:
        yaw, pitch, roll = map(int, match.groups())
        return yaw, pitch, roll
    else:
        raise ValueError(f"Filename pattern not matched: {image_name}")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)   

def train_encoder():    
    
    # Load config
    config = load_config("configs/config_EncoderTrainer.yaml")
    
    # Access hyperparameters values
    trainset_path = config["trainset_path"]
    valset_path = config["valset_path"]    
    
    yaw_min_bin = config["yaw_bins"]["min_bin"]
    yaw_max_bin = config["yaw_bins"]["max_bin"]
    yaw_interval = config["yaw_bins"]["interval"]
    
    pitch_min_bin = config["pitch_bins"]["min_bin"]
    pitch_max_bin = config["pitch_bins"]["max_bin"]
    pitch_interval = config["pitch_bins"]["interval"]
    
    roll_min_bin = config["roll_bins"]["min_bin"]
    roll_max_bin = config["roll_bins"]["max_bin"]
    roll_interval = config["roll_bins"]["interval"]
    
    input_size = config["input_size"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    scheduler_step_size = config["scheduler_step_size"]    
    
    # Load datasets and DataLoaders
    # trainset_path = 'E:/Mahdi/Databases/3D_DB_(50_40_30)_trainset_singleExpressions_240_Subjects_BIWI_3rd_rotation_convention'
    # valset_path = 'E:/Mahdi/Databases/3D_DB_(50_40_30)_valset_singleExpressions_60_Subjects_BIWI_3rd_rotation_convention'
    
    # Check if the saved datasets already exist, if not create them
    if not os.path.exists('data/train_dataset.pth') or not os.path.exists('data/val_dataset.pth'):
        yaw_bins = np.arange(yaw_min_bin, yaw_max_bin, yaw_interval)
        pitch_bins = np.arange(pitch_min_bin, pitch_max_bin, pitch_interval)
        roll_bins = np.arange(roll_min_bin, roll_max_bin, roll_interval)
        
        train_dataset = LandmarkDataset(data_path=trainset_path, yaw_bins=yaw_bins, pitch_bins=pitch_bins, roll_bins=roll_bins)
        val_dataset = LandmarkDataset(data_path=valset_path, yaw_bins=yaw_bins, pitch_bins=pitch_bins, roll_bins=roll_bins)
    
        # Save the datasets for later use
        save_dataset(train_dataset, 'data/train_dataset.pth')
        save_dataset(val_dataset, 'data/val_dataset.pth')
    else:
        # Load the pre-saved datasets
        train_dataset = load_dataset('data/train_dataset.pth')
        val_dataset = load_dataset('data/val_dataset.pth')
    
    # Load the factor matrices and tensor
    loaded_data = np.load('outputs/features/Factor_Matrices.npz')
    U_yaw = loaded_data['U_yaw']
    U_pitch = loaded_data['U_pitch']
    U_roll = loaded_data['U_roll']
    U_id = loaded_data['U_id']
    
    ## enable in case you want to apply penaly to the loss
    # loaded_data = np.load('outputs/features/Trained_data.npz')
    # optimized_yaw = loaded_data['optimized_yaw']
    # optimized_pitch = loaded_data['optimized_pitch']
    # optimized_roll = loaded_data['optimized_roll']
    
    # amplitudes = torch.tensor(optimized_pitch[:,0]).to(device)
    # frequencies = torch.tensor(optimized_pitch[:,1]).to(device)
    # phases = torch.tensor(optimized_pitch[:,2]).to(device)
    # vertical_shifts = torch.tensor(optimized_pitch[:,3]).to(device)
    
    
    # Initializing hyperparameters
    best_val_loss = float('inf')  
    best_train_loss = float('inf')  
    best_model_state = None
    best_epoch = None
    # input_size = 1404  # The number of input to the encoder (This must be equal to the number of flatenned features)
    # batch_size = 256
    # num_epochs = 100
    # learning_rate = 1e-4 # 5e-4 / 1e-4 / 5e-5 / 1e-5
    matrix_dims = [
                    (1, U_yaw.shape[1]),
                    (1, U_pitch.shape[1]),
                    (1, U_roll.shape[1])
                    ]    
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    
    # Initialize model and optimizer
    encoder = LandmarkEncoder(input_size, matrix_dims).to(device)
    
    # Define the optimizer and loss functions
    optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)#, momentum=0.9
    # optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=0.9)
    matrix_loss_fn = nn.MSELoss()
    
    lambda_penalty = 0.1
    
    for epoch in range(num_epochs):
        encoder.train()
            
        epoch_matrix_loss = 0
        epoch_total_loss = 0
        
        # train loop ############################################################################################
        for idx, (x_batch, true_poses) in enumerate(train_loader):  
            
            batch_wy_idx = true_poses[:,0].cpu().numpy()
            batch_wp_idx = true_poses[:,1].cpu().numpy()
            batch_wr_idx = true_poses[:,2].cpu().numpy()
            # batch_U_id = true_poses[:,3].cpu().numpy()
            
            ground_truth_matrices = [
                torch.tensor(U_yaw[batch_wy_idx], dtype=torch.float32).to(device),    # First ground truth matrix
                torch.tensor(U_pitch[batch_wp_idx], dtype=torch.float32).to(device),  # Second ground truth matrix
                torch.tensor(U_roll[batch_wr_idx], dtype=torch.float32).to(device)   # Third ground truth matrix            
            ]
                                  
            batch_landmarks = x_batch.to(device)
            
            # Normalize landmarks
            # mean = batch_landmarks.mean(dim=1, keepdim=True)
            # std = batch_landmarks.std(dim=1, keepdim=True)
            # batch_landmarks = (batch_landmarks - mean) / (std + 1e-6)
            
            # Forward pass
            predicted_matrices = encoder(batch_landmarks)
            # predicted_matrices = encoder(batch_landmarks)
            
            
            # Compute the loss for latent matrices
            matrix_loss = 0
            for pred, gt in zip(predicted_matrices, ground_truth_matrices):
                matrix_loss += matrix_loss_fn(pred, gt.unsqueeze(dim=1))
               
                # penalty = 0
                # if (epoch > num_epochs and epoch % 5 == 0):
                #     penalty = generalized_cosine_penalty(pred, gt.unsqueeze(dim=1), amplitudes, frequencies, phases, vertical_shifts)
    
                        
            # Combine losses        
            total_loss = matrix_loss #+ (lambda_penalty * penalty) 
            
            # Backward pass
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=20.0)
                    
            optimizer.step()  # Optimization step
            
            # Check gradients for NaN
            for name, param in encoder.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients for {name}")
                    break
            
            # Accumulate losses for this batch
            epoch_matrix_loss += matrix_loss.item()
            epoch_total_loss += total_loss.item()
        
        scheduler.step()
        # Print average loss for this epoch
        num_batches = len(train_loader)
        print(f"Training - Epoch [{epoch+1}/{num_epochs}] ------ Total Loss: {epoch_total_loss/num_batches:.4f}")
    
        # Validation ############################################################################################
        # Validation step after each epoch
        encoder.eval()  # Set the model to evaluation mode
        val_matrix_loss = 0
        val_total_loss = 0.0
        val_mae_matrix, val_rmse_matrix = 0.0, 0.0
        
        with torch.no_grad():  # Disable gradient calculation during validation
            # validation loop
            for x_batch, true_poses in val_loader:
                
                batch_wy_idx = true_poses[:,0].cpu().numpy()
                batch_wp_idx = true_poses[:,1].cpu().numpy()
                batch_wr_idx = true_poses[:,2].cpu().numpy()
                # batch_U_id = true_poses[:,3].cpu().numpy()
                
                ground_truth_matrices = [
                    torch.tensor(U_yaw[batch_wy_idx], dtype=torch.float32).to(device),    # First ground truth matrix
                    torch.tensor(U_pitch[batch_wp_idx], dtype=torch.float32).to(device),  # Second ground truth matrix
                    torch.tensor(U_roll[batch_wr_idx], dtype=torch.float32).to(device)    # Third ground truth matrix                 
                ]            
                          
                batch_landmarks = x_batch.to(device)
                
                # Normalize landmarks
                # mean = batch_landmarks.mean(dim=1, keepdim=True)
                # std = batch_landmarks.std(dim=1, keepdim=True)
                # batch_landmarks = (batch_landmarks - mean) / (std + 1e-6)
                
                # Forward pass (same as during training)
                predicted_matrices = encoder(batch_landmarks)
                # predicted_matrices = encoder(batch_landmarks)
    
                # Compute the loss for latent matrices (validation)
                matrix_loss = 0
                batch_mae_matrix, batch_rmse_matrix = 0.0, 0.0
                for pred, gt in zip(predicted_matrices, ground_truth_matrices):
                    matrix_loss += matrix_loss_fn(pred, gt.unsqueeze(dim=1))
                    # penalty = 0
                    # if (epoch > num_epochs and epoch % 5 == 0):
                    #     penalty = generalized_cosine_penalty(pred, gt.unsqueeze(dim=1), amplitudes, frequencies, phases, vertical_shifts)
                    
                    mae, mse, rmse = compute_metrics(pred, gt.unsqueeze(dim=1))
                    batch_mae_matrix += mae
                    batch_rmse_matrix += rmse
                                        
                batch_mae_matrix /= len(predicted_matrices)
                batch_rmse_matrix /= len(predicted_matrices)
                val_mae_matrix += batch_mae_matrix
                val_rmse_matrix += batch_rmse_matrix                                  
                                
                # Combine losses
                total_loss = matrix_loss #+ (lambda_penalty * penalty) 
                
                # Accumulate validation losses
                val_matrix_loss += matrix_loss.item()
                val_total_loss += total_loss.item()
    
    
        val_mae_matrix /= len(val_loader)
        val_rmse_matrix /= len(val_loader)
        
        # Print average validation loss
        val_num_batches = len(val_loader)
        print(f"Validation - Epoch [{epoch+1}/{num_epochs}] ------ Total Loss: {val_total_loss/val_num_batches:.4f} \n")
        print(f"  Predicted Matrices - MAE: {val_mae_matrix:.4f},  RMSE: {val_rmse_matrix:.4f} \n")
        
        # Save the model if the validation loss has improved
        if val_total_loss <= best_val_loss:
            best_val_loss = val_total_loss
            best_train_loss = epoch_total_loss/num_batches
            best_model_state = encoder.state_dict()  # Save the model's state_dict
            best_epoch = epoch + 1
    
    # After training, save the best model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(best_model_state, 'models/Encoder.pth')
    print(f'Best Model Saved at Epoch {best_epoch} with train loss = {best_train_loss:.4f} and val loss = {best_val_loss/val_num_batches:.4f}')

if __name__ == "__main__":   
    
    train_encoder()
    

