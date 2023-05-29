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
    
    m = len(x)
    
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  compute_cost(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}, w = {w}, b = {b}")
        
    return w, b, J_history, w_history #return w and J,w history for graphing

if __name__ == "__main__":
    data = read_data("./data.csv")
    data["km"] /= 10000
    data["price"] /= 10000
    # Create a scatter plot of the data. To change the markers to red "x",
    # we used the 'marker' and 'c' parameters
    plt.scatter(data["km"], data["price"], marker='x', c='r') 

    # Set the title
    plt.title("Price vs. Mileage")
    # Set the y-axis label
    plt.ylabel('Price in $10,000')
    # Set the x-axis label
    plt.xlabel('Mileage in 10,000km')
    plt.show()


    # w, b, J_history, w_history = gradient_descent()
    # PREDICT
    initial_w = 0
    initial_b = 1

    cost = compute_cost(data["km"], data["price"], initial_w, initial_b)
    print(type(cost))
    print(f'Cost at initial w: {cost:.3f}')
    m = data.shape[0]
    # Compute and display gradient with w initialized to zeroes
    initial_w = 0
    initial_b = 0

    tmp_dj_dw, tmp_dj_db = compute_gradient(data["km"], data["price"], initial_w, initial_b)
    print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

    test_w = 0.2
    test_b = 0.2
    tmp_dj_dw, tmp_dj_db = compute_gradient(data["km"], data["price"], test_w, test_b)

    print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)

    initial_w = 0.
    initial_b = 0.

    # gradient descent settings
    iterations = 15000
    alpha = 0.01

    w,b,_,_ = gradient_descent(data["km"], data["price"], initial_w, initial_b, alpha, iterations)
    print("w,b found by gradient descent:", w, b)

    predicted = np.zeros(m)
    for i in range(m):
        predicted[i] = w * data["km"][i] + b


    plt.plot(data["km"], predicted, c = "b")

    # Create a scatter plot of the data. 
    plt.scatter(data["km"], data["price"], marker='x', c='r') 

    # Set the title
    plt.title("Price vs. Mileage")
    # Set the y-axis label
    plt.ylabel('Price in $10,000')
    # Set the x-axis label
    plt.xlabel('Mileage in 10,000km')
    plt.show()

