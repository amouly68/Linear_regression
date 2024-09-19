import numpy as np

def load_model(file="model.npy"):
    return np.load(file)

def load_scaler(file="scalers.npy"):
    data = np.load(file)
    km_min, km_max, price_min, price_max = data
    return km_min, km_max, price_min, price_max

def predict_price(km, theta0, theta1):
    """
    Prédit le prix basé sur le kilométrage en utilisant le modèle de régression linéaire.
    """
    
    km_min, km_max, price_min, price_max = load_scaler()

    km_normalized = (km - km_min) / (km_max - km_min)
    price_normalized = theta0 + (theta1 * km_normalized)
    
    price_denormalized = price_normalized * (price_max - price_min) + price_min

    if isinstance(price_denormalized, np.ndarray):
        price_denormalized = price_denormalized.item()
    if price_denormalized < 0:
        price_denormalized = 0
    
    return price_denormalized

def main():
    theta0, theta1 = load_model()
    
    km = float(input("Entrez le kilométrage de la voiture : "))
    
    prix = predict_price(km, theta0, theta1)
    
    print(f"Le prix estimé pour un kilométrage de {km} km est de {prix:.2f} euros.")

if __name__ == "__main__":
    main()