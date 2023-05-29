import numpy as np
import sys

if __name__ == "__main__":
    data = None
    try:
        data = np.load('predict_res.npz')
    except FileNotFoundError:
        print("Warning: you should start the train program before that! All thetas were set to 0")
    mileage = 0
    while True:
        try:
            mileage = int(input("Enter mileage in km: "))
        except ValueError:
            print("Wrong input")
            continue
        except EOFError:
            print("Bye!")
            sys.exit(0) 
        except KeyboardInterrupt:
            print("Bye!")
            sys.exit(0) 
        except Exception as exc:
            print("Error:", exc)
            continue
        if mileage < 0:
            print("Wrong input")
            continue
        break
    mileage /= 10000
    price = (data['w']*mileage + data['b']) * 10000 if data else 0
    if price >= 0:
        print(f"Predicted price is {price}")
    else:
        print("Error: mileage is too high for the prediction model")
