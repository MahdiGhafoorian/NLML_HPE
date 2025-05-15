# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:29:43 2024

Code to optimize paramteres of estimated angles and identity
This is a non-gradient based optimization
The gradient based optimization blocks are commented out

@author: Mahdi Ghafourian
"""


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch

w_y_values = []
w_p_values = []
w_r_values = []
u_id_values = []
objective_values = []

# Define the cosine function
def func(w, params):
    a, b, c, d = params
    # radian = (w * np.pi) / 180    
    return a * np.cos(b * w + c) + d

# Define the objective function to minimize
def objective(params, W, x, params_y, params_p, params_r):#,   u_id, f_y, f_r):
    w_y, w_p, w_r = params[:3]
    u_id = params[3:]    
    
    # Compute the vectors f_y, f_p, f_r using parameter matrices
    f_y = np.array([func(w_y, p) for p in params_y])
    f_y = f_y.flatten().astype(np.float32)
    
    f_p = np.array([func(w_p, p) for p in params_p])
    f_p = f_p.flatten().astype(np.float32)
    
    f_r = np.array([func(w_r, p) for p in params_r])
    f_r = f_r.flatten().astype(np.float32)
 
    # Compute x^hat
    x_hat = np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, f_p, f_r)    

    # Compute the L2 norm (squared error)
    error = 0.5 * np.sum((x.numpy() - x_hat) ** 2)    
    
    # Store the current w_y and the corresponding objective value
    w_y_values.append(round(w_y, 3))
    w_p_values.append(round(w_p, 3))
    w_r_values.append(round(w_r, 3))
    u_id_values.append([round(i , 3) for i in u_id])
    objective_values.append(round(error, 3))  
        
    return error

def compute_gradient(params, W, x, params_y, params_p, params_r):
    """Compute the gradient of the objective function."""
    w_y, w_p, w_r = params[:3]
    u_id = params[3:]  
    
    # Compute the vectors f_y, f_p, f_r using parameter matrices
    f_y = np.array([func(w_y, p) for p in params_y])
    f_y = f_y.flatten().astype(np.float32)
    
    f_p = np.array([func(w_p, p) for p in params_p])
    f_p = f_p.flatten().astype(np.float32) 
    
    f_r = np.array([func(w_r, p) for p in params_r])
    f_r = f_r.flatten().astype(np.float32)
    
    # Compute the predicted x_hat
    x_hat = np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, f_p, f_r)
    
    # Compute the residuals
    residuals = x.numpy() - x_hat    
    
    # Compute the gradient with respect to w_y
    df_y_dw_y = np.array([-p[0] * p[1] * np.sin(p[1] * w_y + p[2]) for p in params_y])
    df_y_dw_y = df_y_dw_y.flatten().astype(np.float32)
    grad_w_y = -np.sum(residuals * np.einsum('ijklm,i,j,k,l->m', W, u_id, df_y_dw_y, f_p, f_r))
    
    # Compute the gradient with respect to w_p
    df_p_dw_p = np.array([-p[0] * p[1] * np.sin(p[1] * w_p + p[2]) for p in params_p])
    df_p_dw_p = df_p_dw_p.flatten().astype(np.float32)
    grad_w_p = -np.sum(residuals * np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, df_p_dw_p, f_r))
    
    # Compute the gradient with respect to w_r
    df_r_dw_r = np.array([-p[0] * p[1] * np.sin(p[1] * w_r + p[2]) for p in params_r])
    df_r_dw_r = df_r_dw_r.flatten().astype(np.float32)
    grad_w_r = -np.sum(residuals * np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, f_p, df_r_dw_r))   
    
    # Compute the gradient with respect to u_id
    grad_u_id = -np.einsum('ijklm,m->i', W, residuals * np.einsum('ijklm,j,k,l->m', W, f_y, f_p, f_r))

    # Combine all gradients into a single vector
    gradient = np.concatenate(([grad_w_y, grad_w_p, grad_w_r], grad_u_id))

    return gradient    


def func_torch(w, params):
    a, b, c, d = params
    return a * torch.cos(b * w + c) + d

# Define the objective function to minimize
def objective_torch(params, W, x, params_y, params_p, params_r):
    w_y, w_p, w_r = params[:3]
    u_id = params[3:]

    # Compute the vectors f_y, f_p, f_r using parameter matrices
    f_y = torch.stack([func_torch(w_y, p) for p in params_y]).flatten()
    f_p = torch.stack([func_torch(w_p, p) for p in params_p]).flatten()
    f_r = torch.stack([func_torch(w_r, p) for p in params_r]).flatten()

    # Compute x_hat
    x_hat = torch.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, f_p, f_r)

    # Compute the L2 norm (squared error)
    error = 0.5 * torch.sum((x - x_hat) ** 2)

    return error

def optimize_with_sgd(W, x, u_id, u_id_shape, params_y, params_p, params_r, learning_rate=0.001, num_iterations=3000):
    # Initialize parameters with requires_grad=True
    # initial_guess = torch.randn(3 + u_id_shape, dtype=torch.float32, requires_grad=True) 
    initial_guess = torch.zeros(3 + u_id_shape, dtype=torch.float32, requires_grad=True)
    print(f"Initial Guess (requires_grad={initial_guess.requires_grad}):", initial_guess)
    
    for i in range(num_iterations):
        # Reset gradients if they exist
        if initial_guess.grad is not None:
            initial_guess.grad.zero_()
        
        # Calculate loss and perform backward to compute gradients
        loss = objective_torch(initial_guess, W, x, params_y, params_p, params_r)
        
        # Print loss every 200 iterations
        if i % 200 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
        
        loss.backward(retain_graph=True)
        
        # Check if the gradient was calculated correctly
        if initial_guess.grad is not None:
            # Implement gradient clipping
            torch.nn.utils.clip_grad_norm_(initial_guess, max_norm=1.0)  # Clip gradients to prevent explosion
           
            # Apply the gradient update
            with torch.no_grad():
                initial_guess -= learning_rate * initial_guess.grad
        else:
            print(f"Gradient is None at iteration {i}, skipping update.")
            break

    return initial_guess


def Test(W, x, u_id_shape, optimized_params_y, optimized_params_p, optimized_params_r,
         u_id, f_y, f_p, f_r):    
        
    # Initial guesses for w^y, w^p, w^r and u^id (angles set to 0, u^id set to zeros)
    initial_guess = np.zeros(3 + u_id_shape)  # 3 for w^y, w^p, w^r and u_id_shape for u^id

    #-------------------------------- SGD optimization --------------------------------------------
    # Ensure input data is in PyTorch format
    # W = torch.tensor(W, dtype=torch.float32)  # Convert W to PyTorch tensor
    # x = torch.tensor(x, dtype=torch.float32)  # Convert x to PyTorch tensor
    # optimized_params_y = torch.tensor(optimized_params_y, dtype=torch.float32)  # Convert params_y to PyTorch tensor
    # optimized_params_p = torch.tensor(optimized_params_p, dtype=torch.float32)  # Convert params_p to PyTorch tensor
    # optimized_params_r = torch.tensor(optimized_params_r, dtype=torch.float32)  # Convert params_r to PyTorch tensor
    
    # # Example usage
    # optimized_params = optimize_with_sgd(W, x, u_id, u_id_shape, optimized_params_y, optimized_params_p, optimized_params_r)
    
    # optimized_parameters = optimized_params.detach().numpy()
    
        
    # print('Estimated yaw in degree (SGD) = ', np.degrees(optimized_parameters[0]))
    # print('Estimated pitch in degree (SGD) = ', np.degrees(optimized_parameters[1]))
    # print('Estimated roll in degree (SGD) = ', np.degrees(optimized_parameters[2]))
    
    #----------------------------------------------------------------------------------------------
    
    #-------------------------------- Derivative-free optimization --------------------------------
            
    # Perform the optimization
    result = minimize(objective, initial_guess, args=(W, x, optimized_params_y, 
                                                            optimized_params_p, 
                                                            optimized_params_r),#,     u_id, f_y, f_r), 
                      jac=compute_gradient, method='Powell') # Nelder-Mead / L-BFGS-B / Powell 
            
    optimized_params = np.degrees(result.x)
    # Extract the optimized parametersk';[p]
    optimized_est_w_y, optimized_est_w_p, optimized_est_w_r = optimized_params[:3] # result.x[:3]
    optimized_est_u_id = result.x[3:] # r    optimized_params[3:]
    # print(result.x)
    
    #----------------------------------------------------------------------------------------------
   
    #================== plot objective function with different w_p ============
       
    
    # u_id = optimized_est_u_id    
    # w_arr = np.linspace(-100, 100, num=200)
    # # w_arr = np.radians(w_arr)
    # value_arr = np.zeros(200)
    
    # for i in range(len(w_arr)):
        
    #     f_y = np.array([func(np.radians(w_arr[i]), p) for p in optimized_params_y])
    #     f_y = f_y.flatten().astype(np.float32)
        
    #     f_p = np.array([func(np.radians(w_arr[i]), p) for p in optimized_params_p])
    #     f_p = f_p.flatten().astype(np.float32)
        
    #     f_r = np.array([func(np.radians(w_arr[i]), p) for p in optimized_params_r])
    #     f_r = f_r.flatten().astype(np.float32)
    
    #     # Compute x^hat
    #     x_hat = np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, f_p, f_r)
    
    #     # Compute the L2 norm (squared error)
    #     error = 0.5 * np.sum((x.numpy() - x_hat) ** 2)
    #     value_arr[i] = error
    
    # # Find the index of the minimum error
    # min_index = np.argmin(value_arr)
    # min_r = w_arr[min_index]
    # min_value = value_arr[min_index]
    
    # # Increase figure size and resolution
    # plt.figure(figsize=(10, 6), dpi=150)
    
    # # Plot the captured w_y values and the corresponding objective values
    # plt.plot(w_arr, value_arr, marker='.', label='Objective values')
    
    # # Highlight the minimum value with a distinct color and marker
    # plt.plot(min_r, min_value, 'ro', label=f'Minimum value at w_p={min_r:.2f}')

    # # Add labels, title, and legend
    # plt.xlabel('w_p ')
    # plt.ylabel('Objective function value')
    # plt.title('Objective function value vs w_p')
    # plt.legend()
    # plt.show()    
    #==========================================================================
    
    # # Perform optimization    
    # result = minimize(objective, initial_guess, args=(W, x, optimized_params_y,
    #                                                         optimized_params_p, 
    #                                                         optimized_params_r),
    #                   jac=compute_gradient, method='L-BFGS-B')
   
    # # Extract optimized parameters
    # optimized_params = result.x
    # optimized_est_w_y, optimized_est_w_p, optimized_est_w_r = optimized_params[:3]
    # optimized_est_u_id = optimized_params[3:]
    
    # # Output the results
    # print("Optimized angles:")
    # print("w_y:", optimized_est_w_y)
    # print("w_p:", optimized_est_w_p)
    # print("w_r:", optimized_est_w_r)
    
    # # Calculate the error after optimization
    # error_after_optimization = objective(result.x, W, x, optimized_params_y, optimized_params_p, optimized_params_r)
    # # error_after_optimization = objective(result, W, x, optimized_params_y, optimized_params_p, optimized_params_r)
    
    # print(f"Error after optimization: {error_after_optimization}")
    
    # print("\nOptimized u_id vector:")
    # print(optimized_est_u_id)
    
    
    #========================================= min objective =====================================
    # min_initial_guess = np.zeros(3)
    # result2 = minimize(min_objective, min_initial_guess, args=(W, x, optimized_params_y,
    #                                                         optimized_params_p, 
    #                                                         optimized_params_r),#, u_id),
    #                   jac=min_compute_gradient, method='Powell')
    
    # optimized_params = np.degrees(result2.x)
    # optimized_est_w_y, optimized_est_w_p, optimized_est_w_r = optimized_params[:3]
    # optimized_est_u_id = optimized_params[3:]    
    # print(result2.x)
    #=============================================================================================
    return optimized_est_w_y, optimized_est_w_p, optimized_est_w_r, u_id # u_id here should be replaced with optimized_est_u_id


# # Define the objective function to minimize
# def min_objective(params, W, x, params_y, params_p, params_r, u_id):
#     w_y, w_p, w_r = params[:3]

#     # Compute the vectors f_y, f_p, f_r using parameter matrices
#     f_y = np.array([func(w_y, p) for p in params_y])
#     f_y = f_y.flatten().astype(np.float32)
    
#     f_p = np.array([func(w_p, p) for p in params_p])
#     f_p = f_p.flatten().astype(np.float32)
    
#     f_r = np.array([func(w_r, p) for p in params_r])
#     f_r = f_r.flatten().astype(np.float32)
 
#     # Compute x^hat
#     x_hat = np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, f_p, f_r)    

#     # Compute the L2 norm (squared error)
#     error = 0.5 * np.sum((x.numpy() - x_hat) ** 2)
            
#     # Store the current w_y and the corresponding objective value
#     w_y_values.append(round(w_y, 3))
#     w_p_values.append(round(w_p, 3))
#     w_r_values.append(round(w_r, 3))
#     # w_p_values.append(round(w_p[0], 3))
#     objective_values.append(round(error, 3))    
    
#     return error

# def min_compute_gradient(params, W, x, params_y, params_p, params_r, u_id):
#     """Compute the gradient of the objective function."""
#     w_y, w_p, w_r = params[:3]

#     # Compute the vectors f_y, f_p, f_r using parameter matrices
#     f_y = np.array([func(w_y, p) for p in params_y])
#     f_y = f_y.flatten().astype(np.float32)
    
#     f_p = np.array([func(w_p, p) for p in params_p])
#     f_p = f_p.flatten().astype(np.float32) 
    
#     f_r = np.array([func(w_r, p) for p in params_r])
#     f_r = f_r.flatten().astype(np.float32)
    
#     # Compute the predicted x_hat
#     x_hat = np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, f_p, f_r)
    
#     # Compute the residuals
#     residuals = x.numpy() - x_hat
    
#     # Compute the gradient with respect to w_y
#     df_y_dw_y = np.array([-p[0] * p[1] * np.sin(p[1] * w_y + p[2]) for p in params_y])
#     df_y_dw_y = df_y_dw_y.flatten().astype(np.float32)
#     grad_w_y = -np.sum(residuals * np.einsum('ijklm,i,j,k,l->m', W, u_id, df_y_dw_y, f_p, f_r))
    
#     df_p_dw_p = np.array([-p[0] * p[1] * np.sin(p[1] * w_p + p[2]) for p in params_p])
#     df_p_dw_p = df_p_dw_p.flatten().astype(np.float32)
#     grad_w_p = -np.sum(residuals * np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, df_p_dw_p, f_r))

#     df_r_dw_r = np.array([-p[0] * p[1] * np.sin(p[1] * w_r + p[2]) for p in params_r])
#     df_r_dw_r = df_r_dw_r.flatten().astype(np.float32)
#     grad_w_r = -np.sum(residuals * np.einsum('ijklm,i,j,k,l->m', W, u_id, f_y, f_p, df_r_dw_r))    
    
#     # Combine all gradients into a single vector
#     # gradient = grad_w_y #np.concatenate(([grad_w_y, grad_w_p, grad_w_r], grad_u_id))
#     gradient = np.concatenate(([grad_w_y], [grad_w_p], [grad_w_r]))
#     # gradient = grad_w_p

#     return gradient 

