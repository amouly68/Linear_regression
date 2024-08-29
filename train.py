import pandas as pd
import numpy as np



def load_data():
    df = pd.read_csv('data.csv')
    return df['km'].values, df['price'].values

def main():
    """
    This function is the main entry point of the program. It calls the load_data function
    to retrieve the mileage and price data from a CSV file.

    Parameters:
    None

    Returns:
    None
    """
    mileage, price = load_data()

    #set hyperparameter
    learning_rate = 0.01
    num_iterations = 1000

    # Initialize parameters
    m = 0  # slope
    c = 0  # y-intercept
    cost_history = []

    for i in range(num_iterations):
        # Calculate the predicted value
        y_pred = m * mileage + c

        # Calculate the gradient
        dm = (-2/len(mileage)) * np.sum(mileage * (price - y_pred))
        dc = (-2/len(mileage)) * np.sum(price - y_pred)

        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Calculate the cost
        cost = (1/len(mileage)) * np.sum(np.square(price - y_pred))

        # Store the cost in the cost history
        cost_history.append(cost)
    
    print("Final parameters:")
    print("Slope (m):", m)
    print("Y-intercept (c):", c)
    print("Cost history:")
    print(cost_history)
    print("Least Squares Regression Line:")
    print("y =", m, "x +", c)
    
