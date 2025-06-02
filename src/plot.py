import pandas as pd
import matplotlib.pyplot as plt
import os

# Lista analizowanych parametrów
params = ["activation", "batch_size", "learning_rate", "num_layers", "num_neurons"]

# Ścieżka do plików CSV
input_folder = "/Users/filipekmac/Documents/GitHub/classification-model-from-scratch/src/results"
output_folder = "/Users/filipekmac/Documents/GitHub/classification-model-from-scratch/src/plots"

# Utwórz folder, jeśli nie istnieje
os.makedirs(output_folder, exist_ok=True)

def plot_accuracy(df, param_name):
    plt.figure(figsize=(10, 6))
    x = range(len(df))
    labels = df["value"].astype(str)

    # Wykres train/test mean
    plt.plot(x, df["train_mean"], marker='o', label=f"Train (best={df['train_best'].max():.3f})")
    plt.plot(x, df["test_mean"], marker='o', label=f"Test (best={df['test_best'].max():.3f})")

    plt.title(f"Accuracy - {param_name}")
    plt.xlabel("Wartości parametru")
    plt.ylabel("Accuracy")
    plt.xticks(ticks=x, labels=labels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = f"{param_name}_accuracy_combined.png"
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.show()

def plot_metrics(df, param_name):
    plt.figure(figsize=(10, 6))
    x = range(len(df))
    labels = df["value"].astype(str)

    # Wykres prec, rec, f1
    plt.plot(x, df["prec_mean"], marker='o', label="Precision")
    plt.plot(x, df["rec_mean"], marker='o', label="Recall")
    plt.plot(x, df["f1_mean"], marker='o', label="F1-score")

    plt.title(f"Precision / Recall / F1 - {param_name}")
    plt.xlabel("Wartości parametru")
    plt.ylabel("Wartości metryk")
    plt.xticks(ticks=x, labels=labels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = f"{param_name}_metrics_combined.png"
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

    plot_accuracy(df, param)
    plot_metrics(df, param)
