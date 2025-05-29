import pandas as pd
import matplotlib.pyplot as plt
import os

# Lista analizowanych parametrów
params = ["activation", "batch_size", "learning_rate", "num_layers", "num_neurons"]

# Ścieżka do plików CSV
input_folder = "/Users/filipekmac/Documents/GitHub/classification-model-from-scratch/src/results"  # folder z plikami CSV
output_folder = "/Users/filipekmac/Documents/GitHub/classification-model-from-scratch/src/plots"  # folder do zapisu wykresów

# Utwórz folder, jeśli nie istnieje
os.makedirs(output_folder, exist_ok=True)

# Funkcja do rysowania wykresów
def plot_metric(df, param_name, metric_col, title_prefix, ylabel, best_col=None):
    plt.figure(figsize=(10, 6))

    x = range(len(df))
    y = df[metric_col].values
    values = df["value"].astype(str)

    if best_col == "train_best":
        legends = [f"{v} (train_best={round(b,3)})" for v, b in zip(values, df["train_best"])]
    elif best_col == "test_best":
        legends = [f"{v} (test_best={round(b,3)})" for v, b in zip(values, df["test_best"])]
    else:
        legends = values

    for i in range(len(df)):
        plt.plot(x[i], y[i], 'o', markersize=8, label=legends[i])
    plt.plot(x, y, linestyle='-', color='gray', alpha=0.5)

    plt.title(f"{title_prefix} - {param_name}")
    plt.xlabel("Wartości parametru")
    plt.ylabel(ylabel)
    plt.xticks(ticks=x, labels=values)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Zapisz wykres
    plot_filename = f"{param_name}_{metric_col}.png"
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.show()

# Przetwarzanie każdego pliku
for param in params:
    file_path = os.path.join(input_folder, f"results_{param}.csv")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Plik {file_path} nie został znaleziony.")
        continue

    plot_metric(df, param, "train_mean", "Train Accuracy", "Accuracy", best_col="train_best")
    plot_metric(df, param, "test_mean", "Test Accuracy", "Accuracy", best_col="test_best")

    for metric_col, title in zip(["prec_mean", "rec_mean", "f1_mean"], ["Precision", "Recall", "F1-score"]):
        plot_metric(df, param, metric_col, title, title, best_col=None)
