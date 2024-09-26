import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_model(file="model.npy"):
    return np.load(file)

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
        data = pd.read_csv("data.csv")
        km_min, km_max, price_min, price_max = normalise_data(data)
        return km_min, km_max, price_min, price_max


def predict_price(km, theta0, theta1):
    
    km_min, km_max, price_min, price_max = load_scaler()

    km_normalized = (km - km_min) / (km_max - km_min)
    price_normalized = theta0 + (theta1 * km_normalized)
    print(f"Price normalized: {price_normalized}\n")
    
    price_denormalized = price_normalized * (price_max - price_min) + price_min
    # price_denormalized = theta0 + (theta1 * km)


    if isinstance(price_denormalized, np.ndarray):
        price_denormalized = price_denormalized.item()
    if price_denormalized < 0:
        price_denormalized = 0
    
    return price_denormalized

def main():
    try :
        theta0, theta1 = load_model()
        print(f"Les valeurs pour l'estimation: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")
    except FileNotFoundError:
        print("Le modèle n'a pas été trouvé.\nVeuillez l'entrainer avec train.py "
              "avant de faire des prédictions.\nNous continuerons avec theta0 = 0 et theta1 = 0")
        theta0, theta1 = 0, 0 

    try:
        km = float(input("Entrez le kilométrage de la voiture : "))
        if km <= 0:
            print("Le kilométrage doit être supérieur à 0.")
            return 1
        
    except ValueError:
        print("Entrée invalide. Veuillez entrer un nombre valide pour le kilométrage.")
        return 1

    prix = predict_price(km, theta0, theta1)
    print(f"Le prix estimé pour un kilométrage de {km} km est de {prix:.2f} euros.")

if __name__ == "__main__":
    main()
