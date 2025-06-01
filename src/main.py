#  Klasyfikator jakości wina (problem klasyfikacyjny)
#  Sieć neuronowa stworzona całkowicie od zera”
#  Sieć ocenia czy dane wino jest dobre czy nie, więc taka klasyfikacja zero/jedynkowa
#  --------------------------------------------------------------------
#  Nie używamy żadnych wysokopoziomowych bibliotek DL (TensorFlow, PyTorch).
#   Cała logika sieci – warstwy, propagacja w przód i wstecz, optymalizacja,
#   są napisana ręcznie.
#  Wbudowaliśmy pętlę spełniająca wymagania projektu:
#   – co najmniej 5 hiperparametrów,
#   – dla każdego ≥ 4 wartości,
#   – każdy zestaw trenowany ≥ 5 razy,
#   – osobny zapis wyników train/test do plików CSV.
#  --------------------------------------------------------------------
#  UŻYCIE (terminal):
#     python3 src/main.py --file src/data/wine-quality-red.csv --threshold 6 --test_size 0.2 --hidden_layers 2 --hidden_units 32 --activation relu --lr 0.01 --batch 32 --epochs 50 --experiments

import argparsedd
import csv
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

#  FUNKCJE POMOCNICZE: podział danych i standaryzacja
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Losowy podział na zbiory uczący i testowy."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))          # losowa permutacja indeksów
    split = int(len(X) * (1 - test_size))  # miejsce gdzie dane mają być podzielone na 2 części
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def standard_scale(X_train: np.ndarray, X_test: np.ndarray):
		"""Standaryzacja cech metodą Z‑score – średnia i odchylenie liczone tylko z próbek należących do zbioru uczącego, czyli X_train."""
		mean = X_train.mean(axis=0)
		std = X_train.std(axis=0) + 1e-8       # dodajemy małą stałą, by uniknąć dzielenia przez 0
		return (X_train - mean) / std, (X_test - mean) / std

#  FUNKCJE AKTYWACJI + pochodne (potrzebne w backprop)
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU (Rectified Linear Unit)
    Zwraca x, gdy x > 0, w przeciwnym razie 0.
    Wprowadza nieliniowość bez górnego ograniczenia.
    Przyspiesza zbieżność i łagodzi problem zanikania gradientów w głębszych sieciach.
    """
    return np.maximum(0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
    """Pochodna ReLU.
    Gradient wynosi 1 tam, gdzie x > 0, i 0 w pozostałych miejscach.
    Dzięki temu obliczenie jest bardzo szybkie.
    """
    return (x > 0).astype(x.dtype)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh (skalowana hiperboliczna tangens).
    Przyjmuje wartości w przedziale (‑1, 1).
    Symetryczna wokół zera – pomaga, gdy dane wejściowe są również znormalizowane do zera.
    """
    return np.tanh(x)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    """Pochodna funkcji tanh.
    Wzór analityczny: 1 ‑ tanh(x)^2.
    Maksymalny gradient ≈ 1 (w okolicy 0), maleje im |x| większe.
    """
    return 1 - np.tanh(x) ** 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid (funkcja logistyczna).
    Mapuje liczby rzeczywiste na zakres (0, 1) – idealna na warstwę wyjściową binarnej klasyfikacji, którą my tu zastosowaliśmy.
    Charakteryzuje się „spłaszczonymi” końcami, co przy dużych |x| może powodować zanik gradientu (dlatego w ukrytych warstwach częściej używa się ReLU/tanh).
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    """Pochodna sigmoidu.
    Najwygodniej wykorzystać fakt, że s'(x) = s(x)·(1 ‑ s(x)), gdzie s(x) to sam sigmoid.
		Dzięki temu nie liczymy ponownie exp(), co przyspiesza działanie sieci.
    """
    s = sigmoid(x)
    return s * (1 - s)

# słownik, który pozwala nam łatwy wybór aktywacji po nazwie
ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "tanh": (tanh, tanh_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
}

#  DEFINICJA WARSTWY GĘSTEJ (w pełni połączonej)
class DenseLayer:
    """Najprostsza warstwa Dense: z = a_prev · W + b, potem funkcja aktywacji."""
    def __init__(self, input_size: int, output_size: int, activation: str):
        # Inicjalizacja wag z równomiernego rozkładu (±1/√n) – wystarczająca dla MLP.
        limit = 1 / math.sqrt(input_size)
        rng = np.random.default_rng()
        self.W = rng.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros((1, output_size))
        self.act, self.act_grad = ACTIVATIONS[activation]
        # cache na potrzeby backprop, by warstwa działała szybciej
        self.z: np.ndarray | None = None
        self.a_prev: np.ndarray | None = None

    # propagacja w przód - sieć liczy wyjście z danych wejściowych
    def forward(self, a_prev: np.ndarray):
        self.a_prev = a_prev               # zapamiętujemy wejście do gradientów
        self.z = a_prev @ self.W + self.b  # iloczyn + bias
        return self.act(self.z)            # aktywacja nieliniowa

    # propagacja wstecz - wsteczne obliczanie gradientów
    def backward(self, dA: np.ndarray):
        # dA – gradient błędu względem aktywacji wyjściowej tej warstwy
        dz = dA * self.act_grad(self.z)    # łańcuch: dL/dz = dL/dA * dA/dz
        # gradienty wag i biasu – uśredniamy po batchu
        dW = self.a_prev.T @ dz / len(dz)
        db = dz.mean(axis=0, keepdims=True)
        # gradient przekazywany do poprzedniej warstwy
        dA_prev = dz @ self.W.T
        return dA_prev, dW, db

#  STRUKTURA CAŁEJ SIECI (MLP: kilka warst Dense + 1 neuron wyjściowy)
class MLP:
    """Prosty wielowarstwowy perceptron do binarnej klasyfikacji."""
    def __init__(self, input_dim: int, num_layers: int, num_neurons: int,
                 activation: str, learning_rate: float):
        self.layers: list[DenseLayer] = []
        dim_prev = input_dim
        # tworzymy ukryte warstwy
        for _ in range(num_layers):
            self.layers.append(DenseLayer(dim_prev, num_neurons, activation))
            dim_prev = num_neurons
        # pojedynczy neuron wyjściowy + sigmoid
        limit = 1 / math.sqrt(dim_prev)
        rng = np.random.default_rng()
        self.out_W = rng.uniform(-limit, limit, (dim_prev, 1))
        self.out_b = np.zeros((1, 1))
        self.lr = learning_rate

    # propagacja w przód przez całą sieć
    def forward(self, X: np.ndarray):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        z = a @ self.out_W + self.out_b
        y_hat = sigmoid(z)                # prawdopodobieństwo klasy „1”
        self._cache = (a, z)              # zachowujemy w cache, by policzyć gradienty wyjścia
        return y_hat

    # funkcja kosztu (binary CE)
    def compute_loss(self, y_hat: np.ndarray, y_true: np.ndarray):
        eps = 1e-8                        # zapobiega log(0)
        return -np.mean(y_true * np.log(y_hat + eps) +
                        (1 - y_true) * np.log(1 - y_hat + eps))

    # propagacja wstecz
    def backward(self, y_hat: np.ndarray, y_true: np.ndarray):
        # gradient na wyjściu: dL/dz = (y_hat – y_true)/N  (wyprowadzony z BCE)
        a_last, _ = self._cache
        dz_last = (y_hat - y_true) / len(y_true)
        dW_out = a_last.T @ dz_last
        db_out = dz_last.mean(axis=0, keepdims=True)
        dA_prev = dz_last @ self.out_W.T
        # backprop przez ukryte warstwy - cofanie błędu przez kolejne warstwy w odwrotnej kolejności
        grads = []
        for layer in reversed(self.layers):
            dA_prev, dW, db = layer.backward(dA_prev)
            grads.append((layer, dW, db))
        # aktualizacja wag
        self.out_W -= self.lr * dW_out
        self.out_b -= self.lr * db_out
        for layer, dW, db in grads:
            layer.W -= self.lr * dW
            layer.b -= self.lr * db

    # pętla treningowa jednej sieci
    def fit(self, X_train, y_train, epochs: int, batch_size: int):
        n = len(X_train)
        for _ in range(epochs):
						# Tasowanie (mieszanie) kolejności próbek treningowych na początku każdej epoki uczenia
						# Gdy dane są posortowane (np. rosnąca wartość pH), sieć widzi w pierwszych krokach wyłącznie niski zakres cech. To wprowadza stronniczość w początkowych aktualizacjach gradientu.
						# Sieć nie może „nauczyć się” sekwencji próbek, bo w każdym przejściu dostaje je w innej kolejności.
            idx = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]
            # iteracja używając batchy
            for start in range(0, n, batch_size):
                end = start + batch_size
                Xb, yb = X_train[start:end], y_train[start:end]
                y_hat = self.forward(Xb)
                self.backward(y_hat, yb)

    # inferencja: prawdopodobieństwa
    def predict_proba(self, X: np.ndarray):
        return self.forward(X)

#  MIARA SKUTECZNOŚCI – accuracy
def accuracy(y_true: np.ndarray, y_pred_prob: np.ndarray, thr: float = 0.5):
    y_pred = (y_pred_prob >= thr).astype(int)
    return (y_pred == y_true).mean()

#  INNE MIARY – precision, recall, f1
def precision_recall_f1(y_true: np.ndarray, y_pred_prob: np.ndarray, thr=0.5):
    y_pred = (y_pred_prob >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1

#  GŁÓWNA PĘTLA – sprawdza ≥5 hiperparametrów
def run_experiments(X_train, X_test, y_train, y_test,
                    repeats: int = 5, epochs: int = 50):
    """Przeprowadzaamy systematyczne testy hiperparametrów i zapisujemy wyniki."""
    PARAM_GRID = {
        "num_layers":   [1, 2, 3, 4],            # liczba warstw ukrytych
        "num_neurons":  [8, 16, 32, 64],         # neurony w warstwie
        "learning_rate": [0.1, 0.05, 0.01, 0.005], #zmiana wag
        "activation":   ["relu", "tanh", "sigmoid"], #funkcje aktywacji
        "batch_size":   [16, 32, 64, 128], #liczba próbek danych
    }
    
    out_dir = Path("src/results")
    out_dir.mkdir(exist_ok=True)

		# Potrzebujemy jednego punktu odniesienia (baseline), od którego będziemy odchylać tylko jeden hiperparametr naraz
    # Ustalamy „baseline” – zawsze druga wartość z listy, potem zmieniamy 1 parametr naraz
		# PARAM_GRID.items() zwraca pary (nazwa_parametru, lista_wartości), gdzie k to nazwa parametru (np. 'num_layers'), a v[1] to drugi element listy wartości (indeks 1).
		# Dzięki temu baseline nie leży na żadnym z krańców skali, co ułatwia późniejsze porównania.
    BASELINE = {k: v[1] for k, v in PARAM_GRID.items()}

    for param, values in PARAM_GRID.items():
        rows = []
        for v in values:
            cfg = BASELINE.copy(); cfg[param] = v      # tutaj wprowadzamy jedną zmianę parametru naraz w każdej iteracji, tak jak było to wspomniane wyżej
            tr_acc, te_acc, prec_list, recall_list, f1_list = [], [], [], [], []
            # powtarzamy trenowanie, by uśrednić losowość
            for _ in range(repeats):
                model = MLP(
                    input_dim=X_train.shape[1],
                    num_layers=cfg["num_layers"],
                    num_neurons=cfg["num_neurons"],
                    activation=cfg["activation"],
                    learning_rate=cfg["learning_rate"],
                )
                model.fit(X_train, y_train, epochs=epochs, batch_size=cfg["batch_size"])
                tr_acc.append(accuracy(y_train, model.predict_proba(X_train)))
                te_acc.append(accuracy(y_test, model.predict_proba(X_test)))
                p, r, f = precision_recall_f1(y_test, model.predict_proba(X_test))
                prec_list.append(p)
                recall_list.append(r)
                f1_list.append(f)
            row = {
                "value": v,
                "train_mean": np.mean(tr_acc),
                "train_best": np.max(tr_acc),
                "test_mean": np.mean(te_acc),
                "test_best": np.max(te_acc),
                "prec_mean": np.mean(prec_list),
                "rec_mean":  np.mean(recall_list),
                "f1_mean":   np.mean(f1_list),
            }
            rows.append(row)
            print(f"  {param} = {v:<8} | test_mean = {row['test_mean']:.3f}")
        csv_path = out_dir / f"results_{param}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Zapisano wyniki dla  parametru '{param}' do {csv_path}")

# ŁADOWANIE DANYCH
def load_dataset(path: str, threshold: int = 6):
    """Wczytujemy plik CSV oraz binaryzujemy etykietę quality ≥ threshold, tak żeby mieć potem 0/1 zmienną."""
    df = pd.read_csv(path, sep=",")
    df = pd.get_dummies(df, columns=["type"], drop_first=True) # przekształcamy także zmienną 'type'
    X = df.drop(columns=["quality"]).values.astype(np.float32)
    y = (df["quality"] >= threshold).astype(int).values.reshape(-1, 1)
    return X, y

# PARSER ARGUMENTÓW
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Ścieżka do pliku CSV z danymi")
    p.add_argument("--threshold", type=int, default=6, help="Próg jakości dla klasy 1")
    p.add_argument("--test_size", type=float, default=0.2, help="Ułamek danych testowych")
    # hiperparametry
    p.add_argument("--hidden_layers", type=int, default=2)
    p.add_argument("--hidden_units", type=int, default=32)
    p.add_argument("--activation", choices=list(ACTIVATIONS.keys()), default="relu")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--experiments", action="store_true", help="Uruchom siatkę eksperymentów")
    return p.parse_args()

# MAIN
def main():
    args = parse_args()
    print("\nWczytywanie danych")
    X, y = load_dataset(args.file, threshold=args.threshold)
    print(f"Dane: {X.shape[0]} próbek, {X.shape[1]} cech")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)
    X_train, X_test = standard_scale(X_train, X_test)
    print(f"Podział danych: Treningowe: {len(X_train)} | Testowe: {len(X_test)}")

    if args.experiments:
        start = perf_counter()
        run_experiments(X_train, X_test, y_train, y_test, repeats=5, epochs=args.epochs)
        print(f"Cały proces eksperymentów zajął {perf_counter()-start:.1f} sekund")
    else:
        model = MLP(
            input_dim=X_train.shape[1],
            num_layers=args.hidden_layers,
            num_neurons=args.hidden_units,
            activation=args.activation,
            learning_rate=args.lr,
        )
        start = perf_counter()
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch)
        dur = perf_counter() - start
        train_acc = accuracy(y_train, model.predict_proba(X_train))
        test_acc = accuracy(y_test, model.predict_proba(X_test))
        prec, rec, f1 = precision_recall_f1(y_test, model.predict_proba(X_test))
        print(f"\nCzas uczenia: {dur:.1f} s | train_acc = {train_acc:.3f} | test_acc={test_acc:.3f} | prec={prec:.3f} | rec={rec:.3f} | F1={f1:.3f}")

if __name__ == "__main__":
    main()
