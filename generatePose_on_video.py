# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:18:20 2025

@author: Mahdi Ghafourian
"""
# Standard library imports
import math
from math import cos, sin, radians
import time
import argparse

# Third-party library imports
import cv2
import numpy as np
import torch
import mediapipe as mp

# Local application imports
from helpers import FeatureExtractor as FE


def draw_axes(image, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    """
    Draw 3D axes on an image given Euler angles (yaw, pitch, roll).
    Parameters:
        image: The image where axes will be drawn.
        yaw, pitch, roll: Euler angles in degrees.
        tdx, tdy: Translation offsets for the face center.
        size: Length of the axes.
    """
    pitch = radians(pitch)
    yaw = radians(-yaw)  # Invert yaw for correct visualization
    roll = radians(roll)

    # Rotation matrix based on the Euler angles
    R = np.array([
        [cos(yaw) * cos(roll), cos(yaw) * sin(roll), -sin(yaw)],
        [sin(pitch) * sin(yaw) * cos(roll) - cos(pitch) * sin(roll),
         sin(pitch) * sin(yaw) * sin(roll) + cos(pitch) * cos(roll),
         sin(pitch) * cos(yaw)],
        [cos(pitch) * sin(yaw) * cos(roll) + sin(pitch) * sin(roll),
         cos(pitch) * sin(yaw) * sin(roll) - sin(pitch) * cos(roll),
         cos(pitch) * cos(yaw)]
    ])

    # Define the axes in 3D space
    axes = np.array([
        [size, 0, 0],  # X-axis (red)
        [0, size, 0],  # Y-axis (green)
        [0, 0, size]   # Z-axis (blue)
    ])

    # Project the 3D axes to 2D using the rotation matrix
    projected_axes = np.dot(axes, R.T)

    if tdx is None or tdy is None:
        tdx, tdy = image.shape[1] // 2, image.shape[0] // 2

    # Convert 3D points to 2D image plane
    x_axis = (int(tdx), int(tdy)), (int(tdx + projected_axes[0][0]), int(tdy - projected_axes[0][1]))
    y_axis = (int(tdx), int(tdy)), (int(tdx + projected_axes[1][0]), int(tdy - projected_axes[1][1]))
    z_axis = (int(tdx), int(tdy)), (int(tdx + projected_axes[2][0]), int(tdy - projected_axes[2][1]))

    # Draw the axes
    cv2.arrowedLine(image, x_axis[0], x_axis[1], (0, 0, 255), 2, tipLength=0.3)  # Red for X-axis
    cv2.arrowedLine(image, y_axis[0], y_axis[1], (0, 255, 0), 2, tipLength=0.3)  # Green for Y-axis
    cv2.arrowedLine(image, z_axis[0], z_axis[1], (255, 0, 0), 2, tipLength=0.3)  # Blue for Z-axis

    return image


def visualize_axes_on_face(prev_tdx, prev_tdy, MAX_CENTER_JUMP, frame, landmarks, yaw, pitch, roll, size=80):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # Calculate the center of the face by averaging the coordinates of key landmarks
    # We can take landmarks around the eyes and nose as they roughly form the center
    nose_idx = 1  # Typically, the nose is at index 1 in MediaPipe
    left_eye_idx = 33  # Left eye can be a good reference
    right_eye_idx = 263  # Right eye can be a good reference

    # Get the 2D coordinates of the landmarks
    nose = landmarks[nose_idx]
    left_eye = landmarks[left_eye_idx]
    right_eye = landmarks[right_eye_idx]

    # Calculate the center of the face
    new_tdx = (nose.x + left_eye.x + right_eye.x) * frame.shape[1] / 3
    new_tdy = (nose.y + left_eye.y + right_eye.y) * frame.shape[0] / 3
    
    # If first frame, just use the new values
    if prev_tdx is None or prev_tdy is None:
        tdx, tdy = new_tdx, new_tdy
    else:
        # Compute Euclidean distance between previous and new center
        dist = math.sqrt((new_tdx - prev_tdx)**2 + (new_tdy - prev_tdy)**2)        
    
        if dist > MAX_CENTER_JUMP:
            # Use previous stable values
            tdx, tdy = prev_tdx, prev_tdy
        else:
            # Accept new values
            tdx, tdy = new_tdx, new_tdy 

    # X-Axis pointing to right, drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis, drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen), drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    
    # Draw the axes
    cv2.line(frame, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)  # X-Axis (Red)
    cv2.line(frame, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)  # Y-Axis (Green)
    cv2.line(frame, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)  # Z-Axis (Blue)    

    return frame, tdx, tdy


# Video processing
def process_video(source, output_path, model, save_output, device='cpu'):
    """
    Process a video to overlay 3D axes on the detected face.
    Parameters:
        source: Path to the video file or using webcam
        model: Pre-trained PyTorch model for head pose estimation.
        device: Device to run the model ('cpu' or 'cuda').
    """
    prev_tdx, prev_tdy = None, None
    MAX_CENTER_JUMP = 100
        
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) 
    not_detected_lndmrk = 0
    
    # Initialize video capture based on source
    if source == 'webcam':  
        cap = cv2.VideoCapture(0)  
    else:  
        cap = cv2.VideoCapture(source)  
    
    if not cap.isOpened():  
        print("Error: Unable to open video source.")  
        return  
    
    # Get video properties (add default fallback for webcam)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    if fps == 0 or fps != fps:  # NaN or zero
        fps = 30.0  # default fallback
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Frame width
    if width == 0:
        width = 640
        
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    if height == 0:
        height = 480    
    
    # Define the VideoWriter to save the output video if not webcam
    if save_output: 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) 
    else:  
        out = None  
    
    frame_count = 0
    prediction_num = 0
    fps_history = []
    frame_time_history = []
    start_time = time.time()  # Start time for FPS calculation
    
    alpha = 0.4 # Smoothing factor (0 < alpha < 1).

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_start_time = time.time()  # Time at frame start

        # Convert the frame to a PyTorch tensor (no preprocessing required)
        # input_data = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device)  # [C, H, W] and batch dimension
        input_landmarks = FE.get_feature_vector_from_image(face_mesh, frame, normalize=True, isPil=False) 
        
        all_zero = (input_landmarks == 0).all()
        if(all_zero.item()): # if landmarks isn't extracted
            not_detected_lndmrk += 1
            continue
        
        input_landmarks = input_landmarks.unsqueeze(dim=0)
        input_landmarks = input_landmarks.to(device)        
        
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:                
                prediction_num += 1        
                
                with torch.no_grad():
                    predictions = model(input_landmarks)  # Assuming model outputs [yaw, pitch, roll]
                    yaw, pitch, roll = round(np.degrees(predictions[0].item()), 2), round(np.degrees(predictions[1].item()), 2), round(np.degrees(predictions[2].item()), 2)
                
                ####################### Exponential Weighted Average block ####################################################
                
                if prediction_num <= 1:
                    yaw_smoothed = yaw
                    pitch_smoothed = pitch
                    roll_smoothed = roll                    
                else:
                    yaw_smoothed = alpha * yaw + (1-alpha) * yaw_smoothed
                    pitch_smoothed = alpha * pitch + (1-alpha) * pitch_smoothed
                    roll_smoothed = alpha * roll + (1-alpha) * roll_smoothed
                    
                yaw, pitch, roll = yaw_smoothed, pitch_smoothed, roll_smoothed
                
                ###############################################################################################################
                
                # Visualize axes on the face
                frame, prev_tdx, prev_tdy = visualize_axes_on_face(prev_tdx, prev_tdy, MAX_CENTER_JUMP, frame, landmarks.landmark, yaw, pitch, roll)
                
                # Write yaw, pitch, roll values on the frame
                text_yaw = f"Yaw: {yaw:.2f}"
                text_pitch = f"Pitch: {pitch:.2f}"
                text_roll = f"Roll: {roll:.2f}"

                # Add the text to the top-left corner of the frame
                cv2.putText(frame, text_yaw, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, text_pitch, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, text_roll, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        # Calculate FPS dynamically
        frame_time = time.time() - frame_start_time  # Time taken for processing one frame
        frame_time_history.append(frame_time)
        if frame_time > 0:
            current_fps = 1.0 / frame_time  # Instant FPS
        else:
            current_fps = fps  # Default to video FPS if calculation fails
        
        fps_history.append(current_fps)  # Store FPS values
        avg_fps = sum(fps_history) / len(fps_history)  # Compute average FPS

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write the frame with 3D axes to the output video
        if save_output:  
            out.write(frame)

        # Show the frame
        cv2.imshow('Head Pose Estimation', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'average frame processing time = {sum(frame_time_history) / len(frame_time_history) }')
    cap.release()
    if save_output:  
        out.release()
    cv2.destroyAllWindows()
    



if __name__ == "__main__":   
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)  #  'webcam' for realtime HPE using webcam  |  {the path to your video}
    parser.add_argument('--save_output', type=bool, required=True)    # Declare whether you want to save the video or not
    parser.add_argument('--output_path', type=str)  #   the path to save pose included video

    args = parser.parse_args()
    
    # Load the PyTorch model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.jit.load("models/combined_model_scripted_prev.pth", map_location=device)
    model.eval()  # Set the model to evaluation mode
    
    # Path to your video file
    # source = 'webcam'
    source = args.source
    save_output = args.save_output
    if save_output == "tr":
        save_output = True
    elif save_output == "fl":
        save_output = False
        
    
    output_path = args.output_path
    # output_path = 'media/NLML_HPE_demo.mp4'
    
    # Run the video processing
    process_video(source, output_path, model, save_output, device)












