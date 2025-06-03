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
# Dataset class for loading images, extracting landmarks, and providing labels for fine tuning
# class LandmarkDataset(Dataset):
#     def __init__(self, data=None, labels=None, data_path=None, normalize=True, yaw_bins=None, pitch_bins=None, roll_bins=None):
#         self.normalize = normalize
        
#         # Set default bins if none provided
#         self.yaw_bins = yaw_bins if yaw_bins is not None else np.arange(-50, 51, 10).astype(np.float32)
#         self.pitch_bins = pitch_bins if pitch_bins is not None else np.arange(-40, 41, 10).astype(np.float32)
#         self.roll_bins = roll_bins if roll_bins is not None else np.arange(-30, 31, 10).astype(np.float32)
        
#         if data is not None and labels is not None:
#             # Data was preloaded
#             self.data = data
#             self.labels = labels
#         else:
#             self.data = []
#             self.labels = []
#             self.data_path = data_path
#             self._load_data()
            
#         # if self.normalize:
#         #     self._normalize_data()
            
#     def _load_data(self):
#         mp_face_mesh = mp.solutions.face_mesh
#         face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)    
               
#         identities = os.listdir(self.data_path)
#         identities.sort(key=int)
        
#         for U_id, folder in enumerate(identities):
#             print(f'Landmarks of subject {U_id} with name {folder} is readed')
#             folder_path = os.path.join(self.data_path, folder)
#             for image_name in os.listdir(folder_path):
#                 image_path = os.path.join(folder_path, image_name)
#                 try:
#                     # Remove "ID(num1)_" prefix and ".png" suffix, then split to get pose values
#                     main_part = image_name.replace('.png', '').split('_(')[1]  # Extract "(num2)_(num3)_(num4)"
#                     # Remove outer parentheses and split to get wy, wp, wr
#                     wy, wp, wr = map(float, main_part.strip('()').split('_'))
                                        
#                     wy_idx = np.where(self.yaw_bins == wy)[0][0]
#                     wp_idx = np.where(self.pitch_bins == wp)[0][0]
#                     wr_idx = np.where(self.roll_bins == wr)[0][0]
                    
#                     # Extract landmarks
#                     # landmarks = self.extract_landmarks(image_path)
#                     landmarks = FE.get_feature_vector(face_mesh, image_path, normalized=True)
#                     if landmarks is not None:
#                         self.data.append((landmarks.float(), torch.tensor([wy_idx, wp_idx, wr_idx, U_id])))
#                 except (ValueError, IndexError):
#                     print(f"Could not parse pose values from image name: {image_name}")

#     def _normalize_data(self):
        
#         # column_means = torch.mean(self.data, dim=0)
        
#         # # Compute the Euclidean distance between each row and the column mean
#         # distances = torch.norm(self.data - column_means, dim=1)
        
#         # Compute the global min and max across each feature (dimension) across all landmarks
#         min_vals = torch.min(self.data, dim=0)[0]
#         max_vals = torch.max(self.data, dim=0)[0]
        
#         # Avoid division by zero for columns with constant values
#         range_vals = max_vals - min_vals
#         range_vals[range_vals == 0] = 1
        
#         # Normalize to (0, 1)
#         # self.data = self.data - min_vals / range_vals
        
#         # Normalize to (0, 1), then scale to (-1, 1)
#         self.data = 2 * ((self.data - min_vals) / range_vals) - 1
        
#         torch.save(min_vals, 'min_vals.pth')
#         torch.save(max_vals, 'max_vals.pth')
#         pass
       
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        # Check if data is preloaded or newly loaded from images
#        if isinstance(self.data, list):  # Newly loaded
#            landmarks, poses = self.data[idx]
#        else:  # Preloaded tensor data
#            landmarks = self.data[idx]
#            poses = self.labels[idx]
           
#        return landmarks.to(device), poses.to(device)
   
# # Load dataset function
# def load_dataset(file_path):
#     data, labels = torch.load(file_path)
#     # Return as a LandmarkDataset object with pre-loaded data and labels
#     return LandmarkDataset(data=data, labels=labels)


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
    U_id = loaded_data['U_id']
    
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
    
    # Freeze the heads (yaw, pitch, roll networks)
    for param in combined_model.yaw_network.parameters():
        param.requires_grad = False
    for param in combined_model.pitch_network.parameters():
        param.requires_grad = False
    for param in combined_model.roll_network.parameters():
        param.requires_grad = False
    
    # Unfreeze only the encoder
    for param in combined_model.encoder.parameters():
        param.requires_grad = True
    #--------------------------------------------------------------------------------------------
    #---------------------------------------- Single inference ----------------------------------
    
    single_test_path = '3D_DB_(50_40_30)_valset_singleExpressions_180_Subjects/1684/ID1684_(20_10_-30).png'
    
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    
    test_landmarks = FE.get_feature_vector(face_mesh, single_test_path, normalize=True) 
    test_landmarks = test_landmarks.unsqueeze(dim=0)
    test_landmarks = test_landmarks.to(device)
    
    ############ Normalizing the landmark ############
    # Load the saved min and max values from training
    # min_vals = torch.load('min_vals.pth')
    # max_vals = torch.load('max_vals.pth')
    
    # # Compute the range and apply the normalization
    # range_vals = max_vals - min_vals
    # range_vals[range_vals == 0] = 1  # Avoid division by zero
    
    # normalized_test_landmarks = 2 * ((test_landmarks - min_vals) / range_vals) - 1 
    ##################################################
    
    
    
    ######################################################################################################################
    ########################################## Fine tuning using 300W_LP dataset #########################################
    ######################################################################################################################
    # dataset_path = 'C:\\Mahdi\\development\\TokenHPE-main\\datasets\\300W_LP'
    # dataset = 'Pose_300W_LP'
    # filename_list = dataset_path + '\\files.txt'
    
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225])
    
    # transform = transforms.Compose([transforms.Resize(240),
    #                                       transforms.RandomCrop(224),
    #                                       transforms.ToTensor(),
    #                                       normalize])
    
    # pose_dataset = datasets.getDataset(dataset, dataset_path, filename_list, transformations=None, train_mode=False)
    
    # train_size = int(0.8 * len(pose_dataset))  # 80% for training
    # val_size = len(pose_dataset) - train_size  # Remaining for testing
    
    # # Split the dataset
    # train_dataset, val_dataset = random_split(pose_dataset, [train_size, val_size])
    
    # from torch.utils.data import Subset
    # train_dataset = Subset(train_dataset, range(2000))
    # val_dataset = Subset(val_dataset, range(500))
    
    
    # batch_size = 256
    # num_epochs = 100
    # learning_rate = 1e-3
    # not_detected_lndmrk = 0
    
    # # Create DataLoaders for train and val sets
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=0)
    
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0)
    
    # optimizer = optim.SGD(combined_model.encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    # criteria = nn.MSELoss()
    
    # for epoch in range(num_epochs):
    #     combined_model.train()    
    #     epoch_loss = 0
    
    #     for i, (Landmarks, gt_mat, cont_labels, _) in enumerate(train_loader):  
            
    #         batch_landmarks = Landmarks.to(device)
    #         prediction = combined_model(batch_landmarks)
    #         prediction_tensor = torch.cat(prediction, dim=1)
    #         prediction_degrees = torch.round(torch.rad2deg(prediction_tensor), decimals=4)
    #         cont_labels = cont_labels.to(device)
    #         batch_loss = criteria(prediction_degrees, cont_labels)
    #         epoch_loss += batch_loss.item()        
    #         batch_loss.backward()
    #         optimizer.step()
            
    #     scheduler.step()
    #     num_batches = len(train_loader)
    #     print(f"Training - Epoch [{epoch+1}/{num_epochs}] ----- Total Loss: {epoch_loss/num_batches:.4f}")
        
    #     combined_model.eval()  # Set the model to evaluation mode
    #     val_loss = 0
    #     with torch.no_grad():  # Disable gradient calculation during validation
    #         # validation loop
    #         for j, (Landmarks, gt_mat, cont_labels, _) in enumerate(val_loader):  
    #             batch_landmarks = Landmarks.to(device)
    #             prediction = combined_model(batch_landmarks)
    #             prediction_tensor = torch.cat(prediction, dim=1)
    #             prediction_degrees = torch.round(torch.rad2deg(prediction_tensor), decimals=4)
    #             cont_labels = cont_labels.to(device)
    #             batch_loss = criteria(prediction_degrees, cont_labels)
    #             val_loss += batch_loss.item()  
                
    #         val_num_batches = len(val_loader)
    #         print(f"Validation - Epoch [{epoch+1}/{num_epochs}] ----- Total Loss: {val_loss/val_num_batches:.4f} \n")
            
                   
    
    
    ######################################### End of Fine tuning using 300W_LP ###########################################
    ######################################################################################################################
    
    
    ######################################################################################################################
    ######################################## Fine tuning using facescape dataset #########################################
    ######################################################################################################################
    # yaw_bins = np.radians(np.arange(-50, 51, 10).astype(np.float32))
    # pitch_bins = np.radians(np.arange(-40, 41, 10).astype(np.float32))
    # roll_bins = np.radians(np.arange(-30, 31, 10).astype(np.float32))
    
    # batch_size = 256
    # num_epochs = 100
    # learning_rate = 1e-5
    
    # train_dataset = load_dataset('train_dataset.pth')
    # val_dataset = load_dataset('val_dataset.pth')
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    
    # for param in combined_model.encoder.parameters():
    #     param.requires_grad = False
    
    
    # # Define the optimizer and loss functions
    # optimizer = optim.SGD(combined_model.parameters(), lr=learning_rate, weight_decay=1e-5)#, momentum=0.9
    # # optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # # scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    # criteria = nn.MSELoss()
    
    # for epoch in range(num_epochs):
    #     combined_model.train()
        
    #     epoch_matrix_loss = 0
    #     epoch_total_loss = 0
        
    #     # train loop
    #     for idx, (x_batch, true_poses) in enumerate(train_loader):  
            
    #         batch_wy_idx = true_poses[:,0].cpu().numpy()
    #         batch_wp_idx = true_poses[:,1].cpu().numpy()
    #         batch_wr_idx = true_poses[:,2].cpu().numpy()
            
    #         trainset_ground_truth = [
    #             torch.tensor(yaw_bins[batch_wy_idx], dtype=torch.float32).to(device),    # First ground truth matrix
    #             torch.tensor(pitch_bins[batch_wp_idx], dtype=torch.float32).to(device),  # Second ground truth matrix
    #             torch.tensor(roll_bins[batch_wr_idx], dtype=torch.float32).to(device)   # Third ground truth matrix            
    #         ]
            
    #         batch_landmarks = x_batch.to(device)
    #         prediction = combined_model(batch_landmarks)
    #         btch_total_loss = 0
    #         for pred, gt in zip(prediction, trainset_ground_truth):
    #             btch_total_loss += criteria(pred, gt.unsqueeze(dim=1))
    #         btch_total_loss /= len(prediction)
    #         btch_total_loss.backward()
    #         optimizer.step()
    #         epoch_total_loss += btch_total_loss.item()
    #     num_batches = len(train_loader)
    #     print(f"Training - Epoch [{epoch+1}/{num_epochs}] ----- Total Loss: {epoch_total_loss/num_batches:.4f}")
            
    #     combined_model.eval()  # Set the model to evaluation mode
    #     val_total_loss = 0
    #     with torch.no_grad():  # Disable gradient calculation during validation
    #         # validation loop
    #         for x_batch, true_poses in val_loader:
                
    #             batch_wy_idx = true_poses[:,0].cpu().numpy()
    #             batch_wp_idx = true_poses[:,1].cpu().numpy()
    #             batch_wr_idx = true_poses[:,2].cpu().numpy()
                
    #             valset_ground_truth = [
    #                 torch.tensor(yaw_bins[batch_wy_idx], dtype=torch.float32).to(device),    # First ground truth matrix
    #                 torch.tensor(pitch_bins[batch_wp_idx], dtype=torch.float32).to(device),  # Second ground truth matrix
    #                 torch.tensor(roll_bins[batch_wr_idx], dtype=torch.float32).to(device)   # Third ground truth matrix            
    #             ]
                
    #             batch_landmarks = x_batch.to(device)
    #             prediction = combined_model(batch_landmarks)
    #             btch_total_loss = 0
    #             for pred, gt in zip(prediction, valset_ground_truth):
    #                 btch_total_loss += criteria(pred, gt.unsqueeze(dim=1))
    #             btch_total_loss /= len(prediction)
    #             val_total_loss += btch_total_loss.item()
                
    #     val_num_batches = len(val_loader)
    #     print(f"Validation - Epoch [{epoch+1}/{num_epochs}] ----- Total Loss: {val_total_loss/val_num_batches:.4f} \n")
    
    ############################### End of Fine tuning facescape #########################################################
    ######################################################################################################################
        
     
    # Move combined model to the appropriate device
    combined_model.to(device)
    scripted_model = torch.jit.script(combined_model)  # or torch.jit.trace() if model is not dynamic
    scripted_model.save("models/combined_model_scripted.pth")
    combined_model.eval()
    
    # Perform single inspection
    with torch.no_grad():
        predicted_yaw, predicted_pitch, predicted_roll = combined_model(test_landmarks) # yaw_vector, pitch_vector, roll_vector)
    
    print(f"Predicted Yaw: {np.degrees(predicted_yaw.item()):.4f}")
    print(f"Predicted Pitch: {np.degrees(predicted_pitch.item()):.4f}")
    print(f"Predicted Roll: {np.degrees(predicted_roll.item()):.4f}")
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++ Data Inspection block +++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # titles = ['yaw','pitch','roll']  # yaw / pitch / roll
    # titles = ['pitch']
    
    
    ########## Plot Ground Truth data ###################################
    # for titl in titles:
    #     if titl == 'yaw':
    #         bins = np.arange(-50, 51, 10)
    #     elif titl == 'pitch':
    #         bins = np.arange(-40, 41, 10)
    #     elif titl == 'roll':
    #         bins = np.arange(-30, 31, 10)
            
    #     x_sample_deg = np.arange(-180, 181, 5)  # Using intervals of 5 degrees
    #     num_columns = U_pitch.shape[1]
    #     for i in range(num_columns):    
                
    #         # Calculate f(w) for the given parameters for dimension j
    #         bins_arr= np.array(bins)
    #         # f_values = np.array([f(bins_arr[k], a,b,c,d) for k in range(len(bins_arr))])
        
    #         # y values of data
    #         if titl == 'yaw':
    #             vector = U_yaw[:, i] # turn first column of yaw matrix to a vector
    #         elif titl == 'pitch':
    #             vector = U_pitch[:, i]
    #         elif titl == 'roll':
    #             vector = U_roll[:, i]
                
    #         # vec_neg = vector[bins_arr<0]
    #         # vec_pos = vector[bins_arr>=0]
            
    #         # Increase figure size and resolution
    #         plt.figure(figsize=(10, 6), dpi=150)
                    
    #         # plot data of this particular dimension
    #         plt.plot(bins_arr, vector, label='Value for specific angle',
    #                       marker='o', linestyle='--', color='red')
            
            
    #         # Labels and title
    #         plt.xlabel('Angle (degree)')
    #         plt.ylabel('Function value / v')
    #         plt.title(f'Plot of {titl} dimension {i} and f(x) = cos(x)')
            
    #         # Show legend
    #         plt.legend()
    #         plt.grid(True)
    #         # Show the plot
    #         plt.show()
    
    # ############################# Plot Encoder Curves ########################################
    
    # path = '3D_DB_(50_40_30)_valset_singleExpressions_180_Subjects'
    
    # Idd = '1629'
    # PlotEncoderCurves = False
    
    # if PlotEncoderCurves:
    
    #     for title in titles:
    #         test_pth_arr = []
    #         y_coords1 = []
    #         y_coords2 = []
    #         y_coords3 = []
            
    #         gt_coords1 = []
    #         gt_coords2 = []
    #         gt_coords3 = []
            
    #         if (title == 'yaw'):
    #             x_coords = np.arange(-50, 51, 10)
    #             gt_coords1 = U_yaw[:, 0]
    #             gt_coords2 = U_yaw[:, 1]
    #             gt_coords3 = U_yaw[:, 2]
                
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(-50_-40_10).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(-40_-30_20).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(-30_0_10).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(-20_20_20).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(-10_-10_30).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_20).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(10_40_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(20_-40_30).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(30_0_-10).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(40_40_-20).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(50_-20_0).png')
                
    #         elif(title == 'pitch'):
    #             x_coords = np.arange(-40, 41, 10)
    #             gt_coords1 = U_pitch[:, 0]
    #             gt_coords2 = U_pitch[:, 1]
    #             gt_coords3 = U_pitch[:, 2]
                
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_-40_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_-30_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_-20_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_-10_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_10_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_20_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_30_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_40_0).png')
                
    #         elif(title == 'roll'):
    #             x_coords = np.arange(-30, 31, 10)
    #             gt_coords1 = U_roll[:, 0]
    #             gt_coords2 = U_roll[:, 1]
    #             gt_coords3 = U_roll[:, 2]
                
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_-30).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_-20).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_-10).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_0).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_10).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_20).png')
    #             test_pth_arr.append(path+f'/{Idd}/ID{Idd}_(0_0_30).png')
                
            
            
    #         for i in range(len(test_pth_arr)):
    #             test_landmarks = FE.get_feature_vector(face_mesh, test_pth_arr[i], normalized=True) 
    #             test_landmarks = test_landmarks.unsqueeze(dim=0)
    #             test_landmarks = test_landmarks.float()
    #             test_landmarks = test_landmarks.to(device)
    #             # normalized_test_landmarks = 2 * ((test_landmarks - min_vals) / range_vals) - 1  
                
    #             predicted_matrices = encoder(test_landmarks.to(device))
                
    #             if(title == 'yaw'):        
    #                 vector = predicted_matrices[0].flatten().unsqueeze(dim=0)  # yaw_vector
                    
    #             elif(title == 'pitch'):        
    #                 vector = predicted_matrices[1].flatten().unsqueeze(dim=0) # pitch_vector
                    
    #             elif(title == 'roll'):        
    #                 vector = predicted_matrices[2].flatten().unsqueeze(dim=0)  # roll_vector 
                
    #             y_coords1.append(vector[0][0].item())
    #             y_coords2.append(vector[0][1].item())
    #             y_coords3.append(vector[0][2].item())
                
                  
    #             # Plot the points and connect them
    #         plt.figure()
    #         plt.plot(x_coords, y_coords1, marker='o', linestyle='-', color='b', label='Estimated data curve')
    #         plt.plot(x_coords, gt_coords1, label='Ground truth data curve', marker='o', linestyle='--', color='green')  
    #         plt.title(f'First Dimension of {title} Vector')
    #         plt.xlabel('X-axis')
    #         plt.ylabel('Y-axis')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
            
    #         plt.figure()
    #         plt.plot(x_coords, y_coords2, marker='.', linestyle='--', color='r', label='Estimated data curve')
    #         plt.plot(x_coords, gt_coords2, label='Ground truth data curve', marker='o', linestyle='--', color='green')  
    #         plt.title(f'Second Dimension of {title} Vector')
    #         plt.xlabel('X-axis')
    #         plt.ylabel('Y-axis')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
            
    #         plt.figure()
    #         plt.plot(x_coords, y_coords3, marker='*', linestyle='-.', color='black', label='Estimated data curve')
    #         plt.plot(x_coords, gt_coords3, label='Ground truth data curve', marker='o', linestyle='--', color='green')  
    #         plt.title(f'Third Dimension of {title} Vector')
    #         plt.xlabel('X-axis')
    #         plt.ylabel('Y-axis')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
    ############################################################################################
    
    # Pass test landmark to the encoder to get predicted vectors
    # predicted_matrices = encoder(test_landmarks.to(device))
    
    # yaw_vector = predicted_matrices[0].flatten().unsqueeze(dim=0)
    # pitch_vector = predicted_matrices[1].flatten().unsqueeze(dim=0)
    # roll_vector = predicted_matrices[2].flatten().unsqueeze(dim=0)
    
    ############ Normalizing the vectors ############
    # Load the saved min and max values from training
    # min_yaw = torch.tensor(torch.load('min_U_yaw.pth'), dtype=torch.float32).to(device)
    # max_yaw = torch.tensor(torch.load('max_U_yaw.pth'), dtype=torch.float32).to(device)
    
    # yaw_vector = yaw_vector.to(device)
    
    # yaw_vector_normalized = 2 * (yaw_vector - min_yaw) / (max_yaw - min_yaw) - 1
    ##################################################
    
    #+++++++++++++++++++++++++++++++++++++++++++++ End of Inspection block +++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


if __name__ == "__main__":   
    
    model_builder()

