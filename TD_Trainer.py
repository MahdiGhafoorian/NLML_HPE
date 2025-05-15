# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:35:28 2024

Code to optimize paramteres of Training Trignometric functions

@author: Mahdi
"""
import torch
import tensorly as tl
from tensorly import unfold
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit

from functools import partial

# for non-uniform Furier transform 
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt

# Define the function model
# a_j = Amplitude
# b_j = Frequency
# c_j = Phase shift
# d_j = Vertical shift
def func(w_i, a_j, b_j, c_j, d_j):
    # radian = (w_i * 3.14) / 180
    return a_j * np.cos(b_j * w_i + c_j) + d_j

# Define the objective function (L2 norm across all i for a given j)
def objective(params, U_j, w):
    a_j, b_j, c_j, d_j = params
    error_sum = 0
    for i in range(len(U_j)):
        error_sum += (U_j[i] - func(np.radians(w[i]), a_j, b_j, c_j, d_j)) ** 2
    return error_sum/2 

# Define the gradient of the objective function
def objective_grad(params, U_j, w):
    a_j, b_j, c_j, d_j = params
    
    grad_a = 0
    grad_b = 0
    grad_c = 0
    grad_d = 0
    
    for i in range(len(U_j)):
        w_i = np.radians(w[i])
        error = U_j[i] - func(w_i, a_j, b_j, c_j, d_j)
        
        # Compute the gradients
        grad_a -= error * np.cos(b_j * w_i + c_j)
        grad_b -= error * a_j * (-w_i * np.sin(b_j * w_i + c_j))
        grad_c -= error * a_j * (-np.sin(b_j * w_i + c_j))
        grad_d -= error

    return np.array([grad_a, grad_b, grad_c, grad_d])

# Optimization for each j (dimension) across all i (angles)
def optimize_for_matrix(U_matrix, w_vector):
    n_rows, n_cols = U_matrix.shape
    optimized_params = np.zeros((n_cols, 4))  # Store a_j, b_j, c_j, d_j for each j

    for j in range(n_cols):
        U_j = U_matrix[:, j]  # Get the column j (for all i)

        # Initial guess for parameters [a_j, b_j, c_j, d_j]
        initial_guess = [1.0, 1.0, 0.0, 0.0]

        # Optimize parameters for this column j across all i
        result = minimize(objective, initial_guess, args=(U_j, w_vector), method='BFGS')
        
        # Store optimized parameters
        optimized_params[j, :] = result.x
 
    return optimized_params

def gradient_wrapper(params, U_j, w):
    return objective_grad(params, U_j, w)

#do the optimization using computed gradients
def optimize_for_matrix_using_grads(U_matrix, w_vector, initial_guesses): 
    # w_vector is vector of angles
    n_rows, n_cols = U_matrix.shape
    optimized_params = np.zeros((n_cols, 4))  # Store a_j, b_j, c_j, d_j for each j

    for j in range(n_cols):
        U_j = U_matrix[:, j]  # Get the column j (for all i)

        # Initial guess for parameters [a_j, b_j, c_j, d_j] correspond to alpha, beta, gama, phi
        # initial_guess = [2, 0.5, 0.0, 0.0] 
        initial_guess = initial_guesses[j]
        # Optimize parameters for this column j across all i
        result = minimize(
            objective,
            initial_guess,
            args=(U_j, w_vector),
            method='Powell', # COBYLA/ Powell/
            # jac=partial(gradient_wrapper, U_j=U_j, w=w_vector)  # Passing the gradient function
            jac=objective_grad
        )

        # Store optimized parameters
        optimized_params[j, :] = result.x
 
    return optimized_params

#do the optimization using curve fit
def optimize_for_matrix_using_curve_fit(U_matrix, w_vector, initial_guesses): 
    # w_vector is vector of angles    
    n_rows, n_cols = U_matrix.shape
    optimized_params = np.zeros((n_cols, 4))  # Store a_j, b_j, c_j, d_j for each j

    for j in range(n_cols):
        U_j = U_matrix[:, j]  # Get the column j (for all i)

        # Initial guess for parameters [a_j, b_j, c_j, d_j] correspond to alpha, beta, gama, phi
        initial_guess = initial_guesses[j]
        # Use curve_fit to find the best parameters for this column j across all i
        popt, _ = curve_fit(func, np.radians(w_vector), U_j, p0=initial_guess)
           
        # Store optimized parameters
        optimized_params[j, :] = popt
 
    return optimized_params

def cosine_func(w, a, b, c, d):
    return a * torch.cos(b * w + c) + d

#do the optimization using computed gradients
def optimize_for_matrix_using_torch(U_matrix, w_vector, initial_guesses): 
    # w_vector is vector of angles
    n_rows, n_cols = U_matrix.shape
    optimized_params = np.zeros((n_cols, 4))  # Store a_j, b_j, c_j, d_j for each j 

    w_tensor = torch.tensor(np.radians(w_vector), dtype=torch.float32)

    for j in range(n_cols):
        U_j = torch.tensor(U_matrix[:, j], dtype=torch.float32) # Get the column j (for all i)

        # Initial guess for parameters [a_j, b_j, c_j, d_j] correspond to alpha, beta, gama, phi
        initial_guess = initial_guesses[j]
        # Initialize parameters as tensors with gradients
        a = torch.tensor(initial_guess[0], requires_grad=True)
        b = torch.tensor(initial_guess[1], requires_grad=True)
        c = torch.tensor(initial_guess[2], requires_grad=True)
        d = torch.tensor(initial_guess[3], requires_grad=True)
        
        # Define the optimizer with L-BFGS
        optimizer = torch.optim.LBFGS([a, b, c, d], lr=0.1, max_iter=100)
        
        # Define the closure function for the optimizer
        def closure():
            optimizer.zero_grad()
            y_pred = cosine_func(w_tensor, a, b, c, d)
            loss = torch.sum((U_j - y_pred) ** 2)
            loss.backward()
            return loss
        
        # Perform the optimization
        optimizer.step(closure)

        # Store optimized parameters for this column
        optimized_params[j, :] = np.array([a.item(), b.item(), c.item(), d.item()])
 
    return optimized_params  


def est_params_by_nonUniform_Fourier(U, w):
    initial_guess = np.zeros((U.shape[1], 4))
    
    for i in range(U.shape[1]):
        w_radians = np.radians(w)
        theta_nonuniform = np.array(w_radians)
        column_data = U[:, i]

        # Interpolation to a uniform grid
        theta_uniform = np.linspace(min(theta_nonuniform), max(theta_nonuniform), 100)
        column_data_uniform = interp1d(theta_nonuniform, column_data, kind='linear', fill_value='extrapolate')(theta_uniform)
        
        # FFT analysis
        N = len(theta_uniform)
        T = theta_uniform[1] - theta_uniform[0]
        yf = fft(column_data_uniform)
        xf = fftfreq(N, T)
        
        # Dominant frequency, amplitude, phase, and offset
        dominant_freq_index = np.argmax(np.abs(yf))
        dominant_freq = np.abs(xf[dominant_freq_index]) * 2 * np.pi
        dominant_amplitude = np.abs(yf[dominant_freq_index]) / N
        dominant_phase = np.angle(yf[dominant_freq_index])
        offset = np.mean(column_data_uniform)
        
        initial_guess[i] = [dominant_amplitude, dominant_freq, dominant_phase, offset]
    
    return initial_guess

def est_params_by_Uniform_Fourier(U, w):
    initial_guess = np.zeros((U.shape[1], 4))
    w_radians = np.radians(w)
    
    for i in range(U.shape[1]):    
        column_data = U[:, i]
        N = len(column_data)
        freqs = np.fft.fftfreq(N, d=(w_radians[1] - w_radians[0]))
        fft_vals = fft(column_data)
        idx = np.argmax(np.abs(fft_vals[1:])) + 1  # Skip the zero-frequency component
    
        # Estimate b and c from the dominant frequency
        b_initial = 2 * np.pi * np.abs(freqs[idx])  # Convert to angular frequency
        c_initial = np.angle(fft_vals[idx])
    
        # Estimate a as the amplitude of the dominant frequency
        a_initial = 2 * np.abs(fft_vals[idx]) / N
    
        # Estimate d as the mean value of the column
        d_initial = np.mean(column_data)
        
        initial_guess[i] = [a_initial, b_initial, c_initial, d_initial]
    
    return initial_guess

def estimate_init_Fourier_Trans(yaw_params, pitch_params, roll_params):
    U_yaw, w_yaw = yaw_params
    U_pitch, w_pitch = pitch_params
    U_roll, w_roll = roll_params

    initial_guess_yaw = est_params_by_Uniform_Fourier(U_yaw, w_yaw)
    initial_guess_pitch = est_params_by_Uniform_Fourier(U_pitch, w_pitch)
    initial_guess_roll = est_params_by_Uniform_Fourier(U_roll, w_roll)
    #est_params_by_nonUniform_Fourier
    return initial_guess_yaw, initial_guess_pitch, initial_guess_roll       




def est_params_by_Tayor_Series(U, t):
    """
    Estimate alpha, beta, and gamma for any number of y and t values.
    
    Args:
    y: list or array of y values.
    t: list or array of t values.
    
    Returns:
    alpha, beta, gamma: Estimated parameters.
    """
    
    initial_guess = np.zeros((U.shape[1], 4))
    t = np.radians(t)
    
    for i in range(U.shape[1]):
    
        # Ensure y and t are numpy arrays
        y = U[:,i]
    
        # Number of equations (n-1 equations for n points)
        n = len(y)
    
        # Matrix A to hold the coefficients of beta^2 and beta*gamma
        A = np.zeros((n - 1, 2))
        
        # Vector b for the right-hand side of the equations
        b = np.zeros(n - 1)
    
        # Populate A and b based on the given y and t values
        for j in range(n - 1):
            A[j, 0] = (t[j+1]**2 - t[j]**2)  # Coefficients for beta^2
            A[j, 1] = 2 * (t[j+1] - t[j])    # Coefficients for beta*gamma
            b[j] = (y[j] - y[j+1])           # Differences in y values
        
        # Solve for beta^2 and beta*gamma using least squares
        solution = np.linalg.lstsq(A, b / (1 / 2), rcond=None)[0]
    
        # Extract beta^2 and beta*gamma
        beta_squared = abs(solution[0])
        beta_gamma = solution[1]
    
        # Calculate beta and gamma
        beta = np.sqrt(beta_squared)
        gamma = beta_gamma / beta
    
        # Calculate alpha using the first equation
        alpha = 2 * (y[0] - y[1]) / (beta**2 * (t[1]**2 - t[0]**2) + 2 * beta * gamma * (t[1] - t[0]))
        
        phi = [alpha * np.cos(beta * t[j] + gamma) / y[j] for j in range(len(t))]
        phi = np.mean(phi)
        
        initial_guess[i] = [alpha, beta, gamma, phi]

    return initial_guess

def estimate_init_Taylor_Series(yaw_params, pitch_params, roll_params):
    U_yaw, w_yaw = yaw_params
    U_pitch, w_pitch = pitch_params
    U_roll, w_roll = roll_params

    initial_guess_yaw = est_params_by_Tayor_Series(U_yaw, w_yaw)
    initial_guess_pitch = est_params_by_Tayor_Series(U_pitch, w_pitch)
    initial_guess_roll = est_params_by_Tayor_Series(U_roll, w_roll)
    
    return initial_guess_yaw, initial_guess_pitch, initial_guess_roll   


def Train(yaw_params, pitch_params, roll_params):
    
    # each U in these three lines is the factor matrix of yaw, pitch or roll 
    #   and w is a vector of angles (bins) that we had data for in the training set
    U_yaw, w_yaw = yaw_params 
    U_pitch, w_pitch = pitch_params
    U_roll, w_roll = roll_params
    
    # initial guess using Fourier Transform
    initial_guess_yaw, initial_guess_pitch, initial_guess_roll = estimate_init_Fourier_Trans(yaw_params, 
                                                                                        pitch_params, 
                                                                                        roll_params)    
    
    # initial guess using Taylor Series
    initial_guess_yaw2, initial_guess_pitch2, initial_guess_roll2 = estimate_init_Taylor_Series(yaw_params, 
                                                                                        pitch_params, 
                                                                                        roll_params)
    
    #=====================================================================================================
    # ## ploting a column of any factor matrix without optimizing cosine params 
    #=====================================================================================================
    # Function definition
    # def func_cosine(w, a, b, c, d):
    #     return a * np.cos(b * w + c) + d
    
    # factors = (yaw_params[0], pitch_params[0], roll_params[0])
    # # Create the x_sample values from 0 to 360 with intervals of 5 degrees
    # x_sample_deg = np.arange(-180, 181, 5)  # Using intervals of 5 degrees
    # x_sample_rad = np.radians(x_sample_deg)

    # estimated_params = [initial_guess_yaw, initial_guess_pitch, initial_guess_roll2]
    # bins = [yaw_params[1], pitch_params[1], roll_params[1]]
    # titles = ['yaw','pitch','roll']
    
    # for j in range(3):
    #     # bins_arr= np.array(bins[j])
    #     # bins_arr_pos = bins_arr[bins_arr>=0]
    #     # # Convert bins to the range [0, 360]
    #     # bins_arr_neg = bins_arr[bins_arr<0]+360    
    #     num_columns = factors[j].shape[1]
    #     for i in range(num_columns):
            
    #         a, b, c, d = estimated_params[j][i]
            
    #         # Calculate f(x) for all x_sample values
    #         y_sample = func_cosine(x_sample_rad, a,b,c,d)  # np.radians converts degrees to radians
                
    #         # Calculate f(w) for the given parameters for dimension j
    #         bins_arr= np.array(bins[j])
    #         # f_values = np.array([f(bins_arr[k], a,b,c,d) for k in range(len(bins_arr))])
        
    #         # y values of data
    #         vector = factors[j][:, i] # turn first column of yaw matrix to a vector
    #         # vec_neg = vector[bins_arr<0]
    #         # vec_pos = vector[bins_arr>=0]
            
    #         # Increase figure size and resolution
    #         plt.figure(figsize=(10, 6), dpi=150)
            
    #         # Plot the function f(x)
    #         plt.plot(x_sample_deg, y_sample, label='f(x) = cos(x)',marker='.', linestyle='-', color='green')
            
    #         plt.plot(bins_arr, vector, label='Value for specific angle',
    #                      marker='o', linestyle='--', color='blue')
            
    #         # # Plot the values v for the angles k
    #         # plt.plot(bins_arr_pos, vec_pos, label='positive Values for specific angles',
    #         #             marker='o', linestyle='--', color='blue')
            
    #         # # Plot the values v for the angles k
    #         # plt.plot(bins_arr_neg, vec_neg, label='Negative Values for specific angles',
    #         #             marker='o', linestyle='--', color='green')
            
    #         # Labels and title
    #         plt.xlabel('Angle (degree)')
    #         plt.ylabel('Function value / v')
    #         plt.title(f'Plot of {titles[j]} dimension {i} and f(x) = cos(x)')
            
    #         # Show legend
    #         plt.legend()
    #         plt.grid(True)
    #         # Show the plot
    #         plt.show()
    
    #=====================================================================================================
    #=====================================================================================================
    
    optimized_params_yaw = optimize_for_matrix_using_grads(U_yaw, w_yaw, initial_guess_yaw)
    optimized_params_pitch = optimize_for_matrix_using_grads(U_pitch, w_pitch, initial_guess_pitch)
    optimized_params_roll = optimize_for_matrix_using_grads(U_roll, w_roll, initial_guess_roll)    
    ## optimize_for_matrix_using_grads
    # Print results
    print("Optimal parameters for yaw:")
    print(optimized_params_yaw)
    
    print("\nOptimal parameters for pitch:")
    print(optimized_params_pitch)
    
    print("\nOptimal parameters for roll:")
    print(optimized_params_roll)
    
    # from scipy.optimize import curve_fit
    
    # n_cols = U_yaw.shape[1]
    # optimized_params = np.zeros((n_cols, 4)) 
    
    # for col in range(n_cols):  # Iterate over each column
    #     y = U_yaw[:, col]  # Extract one column
    #     initial_guess = [1, 1, 0.0, 0.0] 
    #     params, _ = curve_fit(func, w_yaw, y, p0=initial_guess)
    #     optimized_params[col, :] = params
    #     a_opt, b_opt, c_opt, d_opt = params
    #     print(f"Dimension {col+1}: a={a_opt}, b={b_opt}, c={c_opt}, d={d_opt}")
    
    
    return optimized_params_yaw, optimized_params_pitch, optimized_params_roll
    

