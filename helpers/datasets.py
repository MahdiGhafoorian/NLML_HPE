"""
original code from thohemp:
https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/datasets.py
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset

from PIL import Image, ImageFilter
import utils
import mediapipe as mp
import CreateTensor
import pickle



def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print("dataset path is:", file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

    
# class AFLW2000(Dataset):
#     def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.img_ext = img_ext
#         self.annot_ext = annot_ext
#         filename_list = get_list_from_filenames(filename_path)

#         self.X_train = filename_list
#         self.y_train = filename_list
#         self.image_mode = image_mode
#         self.length = len(filename_list)

#     def __getitem__(self, index):
#         img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))  # For PIL
        
#         img = img.convert(self.image_mode)  # For PIL
#         mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

#         # Crop the face loosely
#         pt2d = utils.get_pt2d_from_mat(mat_path)

#         x_min = min(pt2d[0,:])
#         y_min = min(pt2d[1,:])
#         x_max = max(pt2d[0,:])
#         y_max = max(pt2d[1,:])

#         k = 0.20
#         x_min -= 2 * k * abs(x_max - x_min)
#         y_min -= 2 * k * abs(y_max - y_min)
#         x_max += 2 * k * abs(x_max - x_min)
#         y_max += 0.6 * k * abs(y_max - y_min)
#         # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))  # For PIL
#         img = img[y_min:y_max, x_min:x_max]                                 # For CV2

#         # We get the pose in radians
#         pose = utils.get_ypr_from_mat(mat_path)
#         # And convert to degrees.
#         # no conversion
#         pitch = pose[0]# * 180 / np.pi
#         yaw = pose[1] #* 180 / np.pi
#         roll = pose[2]# * 180 / np.pi
     
#         R = utils.get_R(pitch, yaw, roll) # convert to degrees.

#         labels = torch.FloatTensor([yaw, pitch, roll])


#         if self.transform is not None:
#             img = self.transform(img)

#         return img, torch.FloatTensor(R), labels, self.X_train[index]

#     def __len__(self):
#         # 2,000
#         return self.length

class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform=None, img_ext='.jpg', annot_ext='.mat'):
        self.data_dir = data_dir
        self.transform = transform  # This should handle NumPy arrays directly
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.length = len(filename_list)

    def __getitem__(self, index):
        # Load the image using OpenCV
        img_path = os.path.join(self.data_dir, self.X_train[index] + self.img_ext)
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Image at path {img_path} could not be loaded.")

        # Convert to RGB if needed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load the .mat file and get point coordinates
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        pt2d = utils.get_pt2d_from_mat(mat_path)

        # Crop the face loosely
        x_min = int(min(pt2d[0, :]))
        y_min = int(min(pt2d[1, :]))
        x_max = int(max(pt2d[0, :]))
        y_max = int(max(pt2d[1, :]))

        k = 0.20
        x_min = max(0, int(x_min - 2 * k * abs(x_max - x_min)))
        y_min = max(0, int(y_min - 2 * k * abs(y_max - y_min)))
        x_max = min(img.shape[1], int(x_max + 2 * k * abs(x_max - x_min)))
        y_max = min(img.shape[0], int(y_max + 0.6 * k * abs(y_max - y_min)))

        # Perform the crop
        img = img[y_min:y_max, x_min:x_max]

        # Get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)

        # Extract pitch, yaw, roll
        pitch = pose[0]  # radians
        yaw = pose[1]    # radians
        roll = pose[2]   # radians

        # Convert to rotation matrix
        R = utils.get_R(pitch, yaw, roll)

        # Convert pose to torch tensor
        labels = torch.FloatTensor([yaw, pitch, roll])

        # Apply transformations if provided
        if self.transform is not None:
            img = self.transform(img)  # Expecting transformations that work with NumPy arrays

        return img, torch.FloatTensor(R), labels, self.X_train[index]

    def __len__(self):
        return self.length

class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length

class AFW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]

        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # Around 200
        return self.length

class BIWI(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

        d = np.load(filename_path)

        x_data = d['image']
        print(x_data.shape)
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.X_train[index]))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi
        # cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)


        # Get target tensors
        cont_labels = torch.FloatTensor([pitch, yaw, roll])

        return img, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length

# class Pose_300W_LP(Dataset):  # orignal one that works with pil image
#     # Head pose from 300W-LP dataset
#     def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.img_ext = img_ext
#         self.annot_ext = annot_ext
#         filename_list = get_list_from_filenames(filename_path)

#         self.X_train = filename_list
#         self.y_train = filename_list
#         self.image_mode = image_mode
#         self.length = len(filename_list)

#     def __getitem__(self, index):
#         img = Image.open(os.path.join(
#             self.data_dir, self.X_train[index] + self.img_ext))
#         img = img.convert(self.image_mode)
#         mat_path = os.path.join(
#             self.data_dir, self.y_train[index] + self.annot_ext)

#         # Crop the face loosely
#         pt2d = utils.get_pt2d_from_mat(mat_path)
#         x_min = min(pt2d[0, :])
#         y_min = min(pt2d[1, :])
#         x_max = max(pt2d[0, :])
#         y_max = max(pt2d[1, :])

#         # k = 0.2 to 0.40
#         k = np.random.random_sample() * 0.2 + 0.2
#         x_min -= 0.6 * k * abs(x_max - x_min)
#         y_min -= 2 * k * abs(y_max - y_min)
#         x_max += 0.6 * k * abs(x_max - x_min)
#         y_max += 0.6 * k * abs(y_max - y_min)
#         img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

#         # We get the pose in radians
#         pose = utils.get_ypr_from_mat(mat_path)
#         # And convert to degrees.
#         pitch = pose[0] # * 180 / np.pi
#         yaw = pose[1] #* 180 / np.pi
#         roll = pose[2] # * 180 / np.pi

#         pitch_agl = pose[0] * 180 / np.pi
#         yaw_agl = pose[1] * 180 / np.pi
#         roll_agl = pose[2] * 180 / np.pi
#         # Gray images

#         # Flip?
#         rnd = np.random.random_sample()
#         if rnd < 0.5:
#             yaw = -yaw
#             roll = -roll
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)

#         # Blur?
#         rnd = np.random.random_sample()
#         if rnd < 0.05:
#             img = img.filter(ImageFilter.BLUR)

#         # Add gaussian noise to label
#         #mu, sigma = 0, 0.01 
#         #noise = np.random.normal(mu, sigma, [3,3])
#         #print(noise) 

#         # Get target tensors
#         # Get rotation matrix
#         R = utils.get_R(pitch, yaw, roll)#+ noise
#         cont_labels = torch.FloatTensor([pitch_agl, yaw_agl, roll_agl])
#         # labels = torch.FloatTensor([temp_l_vec, temp_b_vec, temp_f_vec])

#         if self.transform is not None:
#             img = self.transform(img)

#         return img,  torch.FloatTensor(R), cont_labels, self.X_train[index]

#     def __len__(self):
#         # 122,450
#         return self.length


class Pose_300W_LP(Dataset):
    def __init__(self, data_dir, filename_path, transform=None, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.cache_path = "300W_LP_pose_dataset.pkl"
        self.print_interval = 100 
        
        # Initialize MediaPipe FaceMesh 
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, 
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        # Check if cached dataset exists
        if os.path.exists(self.cache_path):
            print("Loading dataset from cache...")
            with open(self.cache_path, "rb") as f:
                self.X_train, self.y_train = pickle.load(f)
        else:
            print("Filtering dataset... This may take a while.")
            # filename_list = get_list_from_filenames(filename_path)
            # Apply pre-filtering
            self.X_train, self.y_train = self.filter_valid_samples(self.X_train, self.y_train)
            
            # Save the filtered dataset for future use
            with open(self.cache_path, "wb") as f:
                pickle.dump((self.X_train, self.y_train), f)
            print(f"Filtering complete. {len(self.X_train)} valid samples found.")
        
        # Apply pre-filtering
        # self.X_train, self.y_train = self.filter_valid_samples(self.X_train, self.y_train)
        self.length = len(self.X_train)
        
        

    def filter_valid_samples(self, X_train, y_train):
        """ Removes samples where face landmarks cannot be detected or angles exceed limits. """
        valid_X, valid_y = [], []
        total = len(X_train)
        
        i = 0
        for x, y in zip(X_train, y_train):
            img_path = os.path.join(self.data_dir, x + self.img_ext)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if image cannot be loaded
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:  # Only keep images with detected landmarks
                mat_path = os.path.join(self.data_dir, y + self.annot_ext)
                pose = utils.get_ypr_from_mat(mat_path)
                pitch, yaw, roll = pose[0], pose[1], pose[2]
                pitch_agl, yaw_agl, roll_agl = pitch * 180 / np.pi, yaw * 180 / np.pi, roll * 180 / np.pi

                # Filter out samples where angles exceed the specified thresholds
                if pitch_agl <= 41 and pitch_agl >=-41 and yaw_agl <= 51 and yaw_agl >= -51 and roll_agl <= 31 and roll_agl >= -31:
                    i += 1
                    valid_X.append(x)
                    valid_y.append(y)
                    # Print progress every `print_interval` images
                    if (i) % self.print_interval == 0 or i + 1 >= total:
                        print(f"Processed {i + 1}/{total} images... {len(valid_X)} valid samples found.")


        print(f"Filtered dataset: {len(valid_X)} valid samples out of {len(X_train)}")
        return valid_X, valid_y

    def __getitem__(self, index):
        
        # Load the image using OpenCV
        img = cv2.imread(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        
        # Convert BGR to RGB (OpenCV loads as BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        rgb_image = np.array(img)
        input_landmarks = CreateTensor.get_feature_vector_from_nparray(self.face_mesh, rgb_image, normalized=True) 

        
        # Convert the image to float32 and normalize [0, 255] to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        pitch = pose[0]
        yaw = pose[1]
        roll = pose[2]

        pitch_agl = pose[0] * 180 / np.pi
        yaw_agl = pose[1] * 180 / np.pi
        roll_agl = pose[2] * 180 / np.pi

        # Flip?
        # rnd = np.random.random_sample()
        # if rnd < 0.5:
        #     yaw = -yaw
        #     roll = -roll
        #     img = np.fliplr(img)

        # Blur?
        # rnd = np.random.random_sample()
        # if rnd < 0.05:
        #     img = cv2.GaussianBlur(img, (5, 5), 0)

        # Get rotation matrix
        R = utils.get_R(pitch, yaw, roll)
        cont_labels = torch.FloatTensor([yaw_agl, pitch_agl, roll_agl])

        # Apply transformations if provided
        # if self.transform is not None:
        #     img = self.transform(img)
        
        
        # Convert numpy array to PyTorch tensor (C, H, W format)
        img = torch.from_numpy(img.copy()).permute(2, 0, 1)  # Make a copy to avoid non-contiguous memory error

        return input_landmarks, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self):
        return self.length 



    
# class Pose_300W_LP(Dataset): # work with cv2
#     def __init__(self, data_dir, filename_path, transform=None, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.img_ext = img_ext
#         self.annot_ext = annot_ext
#         filename_list = get_list_from_filenames(filename_path)

#         self.X_train = filename_list
#         self.y_train = filename_list
#         self.image_mode = image_mode
#         self.length = len(filename_list)

#     def __getitem__(self, index):
#         mp_face_mesh = mp.solutions.face_mesh
#         face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) 

#         # Load the image using OpenCV
#         img = cv2.imread(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        
#         # Convert BGR to RGB (OpenCV loads as BGR by default)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Convert the image to float32 and normalize [0, 255] to [0, 1]
#         img = img.astype(np.float32) / 255.0
        
#         mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

#         # Crop the face loosely
#         pt2d = utils.get_pt2d_from_mat(mat_path)
#         x_min = min(pt2d[0, :])
#         y_min = min(pt2d[1, :])
#         x_max = max(pt2d[0, :])
#         y_max = max(pt2d[1, :])

#         k = np.random.random_sample() * 0.2 + 0.2
#         x_min -= 0.6 * k * abs(x_max - x_min)
#         y_min -= 2 * k * abs(y_max - y_min)
#         x_max += 0.6 * k * abs(x_max - x_min)
#         y_max += 0.6 * k * abs(y_max - y_min)
#         img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

#         # We get the pose in radians
#         pose = utils.get_ypr_from_mat(mat_path)
#         pitch = pose[0]
#         yaw = pose[1]
#         roll = pose[2]

#         pitch_agl = pose[0] * 180 / np.pi
#         yaw_agl = pose[1] * 180 / np.pi
#         roll_agl = pose[2] * 180 / np.pi

#         # Flip?
#         rnd = np.random.random_sample()
#         if rnd < 0.5:
#             yaw = -yaw
#             roll = -roll
#             img = np.fliplr(img)

#         # Blur?
#         rnd = np.random.random_sample()
#         if rnd < 0.05:
#             img = cv2.GaussianBlur(img, (5, 5), 0)

#         # Get rotation matrix
#         R = utils.get_R(pitch, yaw, roll)
#         cont_labels = torch.FloatTensor([yaw_agl, pitch_agl, roll_agl])

#         # Apply transformations if provided
#         # if self.transform is not None:
#         #     img = self.transform(img)
#         rgb_image = np.array(img)
#         input_landmarks = CreateTensor.get_feature_vector_from_nparray(face_mesh, rgb_image, normalized=True) 

#         # Convert numpy array to PyTorch tensor (C, H, W format)
#         # img = torch.from_numpy(img.copy()).permute(2, 0, 1)  # Make a copy to avoid non-contiguous memory error

#         return input_landmarks, torch.FloatTensor(R), cont_labels, self.X_train[index]

#     def __len__(self):
#         return self.length
    
    

def getDataset(dataset, data_dir, filename_list, transformations, train_mode = True):
    if dataset == 'Pose_300W_LP':
            pose_dataset = Pose_300W_LP(
                data_dir, filename_list, transformations)
    elif dataset == 'AFLW2000':
        pose_dataset = AFLW2000(
            data_dir, filename_list, transformations)
    elif dataset == 'BIWI':
        pose_dataset = BIWI(
            data_dir, filename_list, transformations, train_mode= train_mode)
    elif dataset == 'AFLW':
        pose_dataset = AFLW(
            data_dir, filename_list, transformations)
    elif dataset == 'AFW':
        pose_dataset = AFW(
            data_dir, filename_list, transformations)
    else:
        raise NameError('Error: not a valid dataset name')

    return pose_dataset
