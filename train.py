import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math


def _guard_(func):
    def wrapper(*args, **kwargs):
        try:
            return(func(*args, **kwargs))
        except Exception as e:
            print("exception", e) 
            return None
    return wrapper


@_guard_
def read_data(path):
    """Reads data from path, returns pd.DataFame"""
    return pd.read_csv(path)


@_guard_
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) Label
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0] 
    total_cost = 0
    
    for i in range(m):
        total_cost += (w * x[i] + b - y[i]) ** 2

    return total_cost / (2 * m)

@_guard_
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x  - Mileage
      y  - Price
      w, b - Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    """
    
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
  
    for i in range(m):
        y_predict = w * x[i] + b
        dj_dw += (y_predict - y[i]) * x[i]
        dj_db += (y_predict - y[i])
       
    return dj_dw / m, dj_db / m

@_guard_
def gradient_descent(x, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    w_history = []
    b_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Print progress every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            b_history.append(b)
            print(f"Iteration {i:4}: w = {w}, b = {b}")
        
        # check if step is too low
        if (w - w_in) ** 2 < 0.001 * w and (b - b_in) ** 2 < 0.001 * b:
            break
        
    return w, b, w_history, b_history #return w, b and thiers history

if __name__ == "__main__":
    data = read_data("./data.csv")
    data["km"] /= 10000
    data["price"] /= 10000
    # Create a scatter plot of the data. To change the markers to red "x",
    # we used the 'marker' and 'c' parameters
    plt.scatter(data["km"] * 10, data["price"] * 10, marker='x', c='r') 

    # Plotting
    plt.title("Price vs. Mileage")
    plt.ylabel('Price in $1,000')
    plt.xlabel('Mileage in 1,000km')
    plt.show()

    # PREDICT
    m = data.shape[0]
    initial_w = 0.
    initial_b = 0.

    # gradient descent settings
    iterations = 15000
    alpha = 0.01

    w, b, w_hist, b_hist = gradient_descent(data["km"], data["price"], initial_w, initial_b, alpha, iterations)
    print("w,b found by gradient descent:", w, b)

    predicted = np.zeros(m)
    for i in range(m):
        predicted[i] = w * data["km"][i] + b

    # Plotting
    plt.plot(data["km"] * 10, predicted * 10, c = "b")
    plt.scatter(data["km"] * 10, data["price"] * 10, marker='x', c='r') 
    plt.title("Price vs. Mileage")
    plt.ylabel('Price in $1,000')
    plt.xlabel('Mileage in 1,000km')
    plt.show()
    
    # Save result
    

