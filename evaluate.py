import numpy as np
import pandas as pd
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
def mAbsErr(data, w, b, k_norm):
    result = 0
    m = data.shape[0]
    for i in range(m):
        price = w * data['km'][i] + b * k_norm
        result += abs(price - data['price'][i])
    return result / m


@_guard_
def mSqrErr(data, w, b, k_norm):
    result = 0
    m = data.shape[0]
    for i in range(m):
        price = w * data['km'][i] + b * k_norm
        result += (price - data['price'][i]) ** 2
    return result / m


@_guard_
def mSqrtErr(data, w, b, k_norm):
    return math.sqrt(mSqrErr(data, w, b, k_norm))


@_guard_
def r2score(data, w, b, k_norm):
    rss = 0
    m = data.shape[0]
    for i in range(m):
        price = w * data['km'][i] + b * k_norm
        rss += (price - data['price'][i]) ** 2
    tss = 0
    avg = np.average(data['price'])
    for i in range(m):
        tss += (avg - data['price'][i]) ** 2
    return 1 - rss / tss


if __name__ == "__main__":
    data = None
    predicted = None
    try:
        predicted = np.load('predict_res.npz')
        w = predicted["w"]
        b = predicted["b"]
        k_norm = predicted["k_norm"]
    except FileNotFoundError:
        print("Error: you should start the train program before that!")
    try:
        data = read_data('./data.csv')
    except FileNotFoundError:
        print("Error: data file is not found")
    
    if data is not None and predicted is not None:
        print(f"Mean Absolute Error: ${mAbsErr(data, w, b, k_norm)}")
        print(f"Mean Squared Error: {mSqrErr(data, w, b, k_norm)}($**2)")
        print(f"Root Mean Squared Error: ${mSqrtErr(data, w, b, k_norm)}")
        print(f"R2 Score: {r2score(data, w, b, k_norm)}%")
