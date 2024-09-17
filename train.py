import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import threading

def load_scaler(file="scalers.npy"):
    data = np.load(file)
    km_min, km_max, price_min, price_max = data
    return km_min, km_max, price_min, price_max


def save_scalers(scaler_km, scaler_price, filename="scalers.npy"):
    np.save(filename, np.array([scaler_km.data_min_, scaler_km.data_max_, 
                                scaler_price.data_min_, scaler_price.data_max_]))

def load_data(filename):
    data = pd.read_csv("data.csv")
    scaler_km = MinMaxScaler()
    scaler_price = MinMaxScaler()

    # Normaliser les colonnes 'mileage' et 'price'
    data[['km']] = scaler_km.fit_transform(data[['km']])
    data[['price']] = scaler_price.fit_transform(data[['price']])

    save_scalers(scaler_km, scaler_price)
    
    # scaler = MinMaxScaler()
    # data[['km', 'price']] = scaler.fit_transform(data[['km', 'price']])
    return data['km'].values, data['price'].values

def train_model(km, price, learning_rate, num_iterations):
    """
    Train the model by adjusting theta0 and theta1 using gradient descent.
    """
    m = len(km)  # Number of training examples
    print (f"length of list of training examples: {m}")
    theta0 = 0
    theta1 = 0
    
    for _ in range(num_iterations):
        estimate = theta0 + theta1 * km
        error = estimate - price
        
        # Calculate the new values for theta0 and theta1
        tmp_theta0 = theta0 - (learning_rate * np.sum(error) / m)
        tmp_theta1 = theta1 - (learning_rate * np.sum(error * km) / m)

        theta0 = tmp_theta0
        theta1 = tmp_theta1
    
    return theta0, theta1

def save_model(theta0, theta1, filename="model.npy"):
    np.save(filename, np.array([theta0, theta1]))

# def plot_graph(km, price, theta0, theta1):
#     """
#     Display the scatter plot of data and the linear regression line.
#     """
    
#     plt.scatter(km, price, color='blue', label='Real Data')
#     estimate = theta0 + theta1 * km
#     plt.plot(km, estimate, color='red', label='Linear Regression')
#     plt.xlabel("km")
#     plt.ylabel("Price")
#     plt.title("Linear Regression: Price vs km")
#     plt.legend()
#     plt.show()


def plot_graph(km, price, theta0, theta1):

    km_min, km_max, price_min, price_max = load_scaler()

    km_denorm = km * (km_max - km_min) + km_min
    price_denorm = price * (price_max - price_min) + price_min

    # Afficher le nuage de points des données réelles dénormalisées
    plt.scatter(km_denorm, price_denorm, color='blue', label='Data')

    # predicted_price_norm = theta0 + theta1 * km_denorm
    # predicted_price_denorm = predicted_price_norm * (price_max - price_min) + price_min
    # plt.plot(km_denorm, predicted_price_denorm, color='red', label='Régression linéaire')

    # Calculer et afficher la ligne de régression dénormalisée
    km_range = np.linspace(min(km), max(km), 100)
    km_range_denorm = km_range * (km_max - km_min) + km_min
    predicted_price_norm = theta0 + theta1 * km_range
    predicted_price_denorm = predicted_price_norm * (price_max - price_min) + price_min

    plt.plot(km_range_denorm, predicted_price_denorm, color='red', label='Régression linéaire')


    # Ajouter les labels et la légende
    plt.xlabel("Kilométrage (km)")
    plt.ylabel("Prix (euros)")
    plt.title("Régression Linéaire : Prix vs Kilométrage ")
    plt.legend()
    plt.show()

def main():
    # Load the data
    km, price = load_data("data.csv")
    
    # Set hyperparameters
    learning_rate = 0.98
    num_iterations = 100
    
    # Train the model
    theta0, theta1 = train_model(km, price, learning_rate, num_iterations)
    
    # Save the model
    save_model(theta0, theta1)
    
    # Display the results
    print(f"Trained model: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")
    
    # Plot the graph
    plot_graph(km, price, theta0, theta1)

if __name__ == "__main__":
    main()

    
