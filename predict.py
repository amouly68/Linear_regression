import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

def load_model(file="model.npy"):
    try:
        return np.load(file)
    except FileNotFoundError:
        print("Modèle non trouvé, utilisation de theta0 = 0 et theta1 = 0.")
        return 0, 0

def normalise_data(data):
    scaler_km = MinMaxScaler()
    scaler_price = MinMaxScaler()

    data[['km']] = scaler_km.fit_transform(data[['km']])
    data[['price']] = scaler_price.fit_transform(data[['price']])

    return (scaler_km.data_min_, scaler_km.data_max_, scaler_price.data_min_, scaler_price.data_max_)
    

def load_scaler(file="scalers.npy"):
    try:
        data = np.load(file)
        km_min, km_max, price_min, price_max = data
        return km_min, km_max, price_min, price_max
    except FileNotFoundError:
        print("Scalers non trouvés, recalcul à partir du fichier de données.")
        data = load_data()
        return normalise_data(data)
    
def load_data():
    try:
        data = pd.read_csv("data.csv")
        
    except FileNotFoundError:
        print("Fichier 'data.csv' introuvable.")
        sys.exit(1)
    return data


def predict_price(km, theta0, theta1):
    
    km_min, km_max, price_min, price_max = load_scaler()

    km_normalized = (km - km_min) / (km_max - km_min)
    price_normalized = theta0 + (theta1 * km_normalized)
    
    price_denormalized = price_normalized * (price_max - price_min) + price_min


    if isinstance(price_denormalized, np.ndarray):
        price_denormalized = price_denormalized.item()
    if isinstance(price_normalized, np.ndarray):
        price_normalized = price_normalized.item()

    if theta0 == 0 and theta1 == 0:
        return price_normalized
    return max(price_denormalized, 0)

def main():
    theta0, theta1 = load_model()

    try:
        km = float(input("Entrez le kilométrage de la voiture : "))
        if km < 0:
            print("Le kilométrage doit être supérieur à 0.")
            return 1
    except ValueError:
        print("Entrée invalide. Veuillez entrer un nombre valide pour le kilométrage.")
        return 1

    prix = predict_price(km, theta0, theta1)

    print(f"Le prix estimé pour un kilométrage de {km} km est de {prix:.2f} euros.")

if __name__ == "__main__":
    main()
