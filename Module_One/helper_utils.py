import torch
import matplotlib.pyplot as plt

def to_plot_array(t):
    return t.detach().cpu().numpy().reshape(-1)


def plot_results(model, distances, times):
    
    """
    Plots the actual data points and the model's predicted line for a given dataset.

    Args:
        model: The trained machine learning model to use for predictions.
        distances: The input data points (features) for the model.
        times: The target data points (labels) for the plot.
    """
    model.eval()

    with torch.no_grad():
        predicted_times = model(distances)

    x = to_plot_array(distances)
    y = to_plot_array(times)
    y_pred = to_plot_array(predicted_times)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y, color="orange", marker="o", linestyle="None",
            label="Actual Delivery Times")
    ax.plot(x, y_pred, color="green", linestyle="-",
            label="Predicted Line")

    ax.set_title("Actual vs. Predicted Delivery Times")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Time (minutes)")
    ax.legend()
    ax.grid(True)

    plt.show()
    plt.close(fig)




    
def plot_nonlinear_comparison(model, new_distances, new_times):
    model.eval()

    with torch.no_grad():
        predictions = model(new_distances)

    x = to_plot_array(new_distances)
    y = to_plot_array(new_times)
    y_pred = to_plot_array(predictions)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y, color="orange", marker="o", linestyle="None",
            label="Actual Data (Bikes & Cars)")
    ax.plot(x, y_pred, color="green", linestyle="-",
            label="Linear Model Predictions")

    ax.set_title("Linear Model vs. Non-Linear Reality")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Time (minutes)")
    ax.legend()
    ax.grid(True)

    plt.show()
    plt.close(fig)