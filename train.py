import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.animation import FuncAnimation


def load_scaler(file="scalers.npy"):
    data = np.load(file)
    km_min, km_max, price_min, price_max = data
    return km_min, km_max, price_min, price_max



def normalise_data(data):
    scaler_km = MinMaxScaler()
    scaler_price = MinMaxScaler()

    data[['km']] = scaler_km.fit_transform(data[['km']])
    data[['price']] = scaler_price.fit_transform(data[['price']])

    np.save("scalers.npy", np.array([scaler_km.data_min_, scaler_km.data_max_, 
                                scaler_price.data_min_, scaler_price.data_max_]))
    
    return data['km'].values, data['price'].values


def train_model_with_tracking(km, price, learning_rate, num_iterations):
    m = len(km)
    theta0 = 0
    theta1 = 0

    interval = max(num_iterations // 10, 1)
    
    theta0_history = []
    theta1_history = []
    
    for i in range(1, num_iterations + 1):
        estimate = theta0 + theta1 * km
        error = estimate - price
        
        tmp_theta0 = theta0 - (learning_rate * np.sum(error) / m)
        tmp_theta1 = theta1 - (learning_rate * np.sum(error * km) / m)

        theta0 = tmp_theta0
        theta1 = tmp_theta1

        if i % interval == 0:
            theta0_history.append(theta0)
            theta1_history.append(theta1)

    theta0_history.append(theta0)
    theta1_history.append(theta1)    

    return theta0, theta1, theta0_history, theta1_history


def save_model(theta0, theta1, filename="model.npy"):

    print(f"Trained model: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")
    np.save(filename, np.array([theta0, theta1]))


def plot_graph(km, price, theta0, theta1):
    km_min, km_max, price_min, price_max = load_scaler()

    km_denorm = km * (km_max - km_min) + km_min
    price_denorm = price * (price_max - price_min) + price_min

    plt.scatter(km_denorm, price_denorm, color='blue', label='Data')

    km_range = np.linspace(min(km), max(km), 100)
    km_range_denorm = km_range * (km_max - km_min) + km_min
    predicted_price_norm = theta0 + theta1 * km_range
    predicted_price_denorm = predicted_price_norm * (price_max - price_min) + price_min

    plt.plot(km_range_denorm, predicted_price_denorm, color='red', label='Régression linéaire')

    plt.xlabel("Kilométrage (km)")
    plt.ylabel("Prix (euros)")
    plt.title("Régression Linéaire : Prix vs Kilométrage")
    plt.legend()
    plt.show(block=False)


def plot_regression_evolution(km, price, theta0_history, theta1_history, num_steps=100):
    km_min, km_max, price_min, price_max = load_scaler()
    km_denorm = km * (km_max - km_min) + km_min
    price_denorm = price * (price_max - price_min) + price_min

    plt.scatter(km_denorm, price_denorm, color='blue', label='Data')

    for i in range(0, len(theta0_history), num_steps):
        theta0 = theta0_history[i]
        theta1 = theta1_history[i]
        price_predicted = theta0 + theta1 * km
        price_predicted_denorm = price_predicted * (price_max - price_min) + price_min
        plt.plot(km_denorm, price_predicted_denorm, label=f'Iteration {i}')
    
    plt.xlabel("Kilométrage (km)")
    plt.ylabel("Prix (euros)")
    plt.title("Evolution of Linear Regression Lines")
    plt.legend()
    plt.show()


def animate_regression_evolution(km, price, theta0_history, theta1_history, num_iterations, interval=1500):
    km_min, km_max, price_min, price_max = load_scaler()

    km_denorm = km * (km_max - km_min) + km_min
    price_denorm = price * (price_max - price_min) + price_min

    fig, ax = plt.subplots()

    ax.scatter(km_denorm, price_denorm, color='blue', label='Data')

    ax.set_xlim([0, km_denorm.max() * 1.1])
    ax.set_ylim([3000, price_denorm.max() * 1.1])

    line, = ax.plot([], [], color='red', label='Régression linéaire')

    iteration_text = ax.text(0.60, 0.82, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    theta0_text = ax.text(0.50, 0.77, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    theta1_text = ax.text(0.50, 0.72, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    def init():
        line.set_data([], [])
        iteration_text.set_text('')
        return line, iteration_text

    def update(i):
        theta0 = theta0_history[i]
        theta1 = theta1_history[i]
        
        km_range = np.linspace(min(km), max(km), 100)
        km_range_denorm = km_range * (km_max - km_min) + km_min
        predicted_price_norm = theta0 + theta1 * km_range
        predicted_price_denorm = predicted_price_norm * (price_max - price_min) + price_min

        line.set_data(km_range_denorm, predicted_price_denorm)
        iter = i / 10 * num_iterations
        iteration_text.set_text(f'Itération : {iter:.0f}')
        theta0_text.set_text(f'theta0 : {theta0:.12f}')
        theta1_text.set_text(f'theta1 : {theta1:.12f}')
        return line, iteration_text, theta1_text, theta0_text

    ani = FuncAnimation(fig, update, frames=len(theta0_history), init_func=init,
                        blit=True, interval=interval, repeat=False)

    plt.xlabel("Kilométrage (km)")
    plt.ylabel("Prix (euros)")
    plt.title("Evolution de la Régression Linéaire")
    plt.legend()
    plt.show()


def evaluate_error(true_price, km, theta0, theta1):
    estimate = theta0 + theta1 * km
    estimate_denorm = estimate * (true_price.max() - true_price.min()) + true_price.min()
    error = estimate_denorm - true_price

    mse = np.mean(error ** 2)
    print(f"Mean Squared Error (MSE) : {mse:.4f}")

    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")

    mae = np.mean(np.abs(error))
    print(f"Mean Absolute Error (MAE) : {mae:.4f}")

    ss_residual = np.sum(error ** 2)
    ss_total = np.sum((true_price - np.mean(true_price)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"R-squared : {r_squared:.4f}")


def fine_tune_hyperparameters(km, price, learning_rates, iterations_list):
    best_mse = float('inf')
    best_theta0 = None
    best_theta1 = None
    best_params = None

    for learning_rate in learning_rates:
        for num_iterations in iterations_list:
            theta0, theta1, _, _ = train_model_with_tracking(km, price, learning_rate, num_iterations)
            
            estimate = theta0 + theta1 * km
            mse = np.mean((estimate - price) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_theta0 = theta0
                best_theta1 = theta1
                best_params = (learning_rate, num_iterations)
    
    print(f"Best learning_rate: {best_params[0]}, Best num_iterations: {best_params[1]}")
    print(f"Best theta0: {best_theta0}, Best theta1: {best_theta1}, Best MSE: {best_mse}")
    return best_params


def main():
    
    try:
        data = pd.read_csv("data.csv")
    except FileNotFoundError:
        print("Fichier 'data.csv' introuvable.")
        return 1
    true_price = data['price'].values
    km, price = normalise_data(data)
    
    # learning_rates = [0.001, 0.01, 0.1, 0.3]
    # iterations_list = [500, 1000, 2000]
    # best_params = fine_tune_hyperparameters(km, price, learning_rates, iterations_list)

    learning_rate = 0.3
    num_iterations = 200
    
    # theta0, theta1, theta0_history, theta1_history = train_model_with_tracking(km, price, best_params[0], best_params[1])
    theta0, theta1, theta0_history, theta1_history = train_model_with_tracking(km, price, learning_rate, num_iterations)

    save_model(theta0, theta1)

    evaluate_error(true_price, km, theta0, theta1)
    
    # plot_regression_evolution(km, price, theta0_history, theta1_history, num_steps=100)
    # plot_graph(km, price, theta0, theta1)

    # animate_regression_evolution(km, price, theta0_history, theta1_history, best_params[1])
    animate_regression_evolution(km, price, theta0_history, theta1_history, num_iterations)
    

if __name__ == "__main__":
    main()
