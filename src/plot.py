import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Ścieżka do folderu z plikami CSV
csv_folder = "/Users/filipekmac/Documents/GitHub/classification-model-from-scratch/src/results"  # <-- ZMIEŃ to na rzeczywistą ścieżkę

# Wczytaj wszystkie pliki CSV z folderu
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

# Przetwórz każdy plik
for file in csv_files:
    df = pd.read_csv(file)
    base_name = os.path.splitext(os.path.basename(file))[0]

    # Ustawienia ogólne
    x = df['value']

    # Tworzenie wykresów
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Wyniki: {base_name}", fontsize=16)

    # 1. Accuracy
    axs[0, 0].plot(x, df['train_mean'], label='Train Accuracy', marker='o')
    axs[0, 0].plot(x, df['test_mean'], label='Test Accuracy', marker='x')
    axs[0, 0].set_title("Accuracy (Mean)")
    axs[0, 0].set_xlabel("Value")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Precision
    axs[0, 1].plot(x, df['prec_mean'], label='Precision', color='green', marker='s')
    axs[0, 1].set_title("Precision")
    axs[0, 1].set_xlabel("Value")
    axs[0, 1].set_ylabel("Precision")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Recall
    axs[1, 0].plot(x, df['rec_mean'], label='Recall', color='orange', marker='^')
    axs[1, 0].set_title("Recall")
    axs[1, 0].set_xlabel("Value")
    axs[1, 0].set_ylabel("Recall")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. F1-score
    axs[1, 1].plot(x, df['f1_mean'], label='F1 Score', color='purple', marker='D')
    axs[1, 1].set_title("F1 Score")
    axs[1, 1].set_xlabel("Value")
    axs[1, 1].set_ylabel("F1 Score")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    # lub zapisz:
    # plt.savefig(f"{base_name}_metrics.png")
