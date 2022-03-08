from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

def plot_history(history: History, variable_name: str):
    history_dict = history.history

    train_values = history_dict[variable_name]
    val_values = history_dict[f"val_{variable_name}"]
    epochs = range(1, len(train_values) + 1)

    plt.figure()
    plt.plot(epochs, train_values, "bo", label=f"Training {variable_name}")
    plt.plot(epochs, val_values, "b", label=f"Validation {variable_name}")
    plt.title(f"Training and validation {variable_name}")
    plt.xlabel("Epochs")
    plt.ylabel(f"{variable_name}")
    plt.legend()
    plt.show()
