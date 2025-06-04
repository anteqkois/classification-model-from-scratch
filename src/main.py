# Biblioteki potrzebne do wywołania skryptu
import argparse
import csv
import math
from pathlib import Path
from time import perf_counter
import sys
import numpy as np
import pandas as pd

            ### FUNKCJE POMOCNICZNE ###
"""
train_test_split() - realizuje losowy podział na zbiór treningowy i testowy, działa podobnie do funkcji train_test_split() z biblioteki sklearn
X: dane wejściowe
y: etykiety (wartości docelowe 0 lub 1)
test_size: proporcja danych przeznaczonych do testu
seed: ziarno generatora liczb losowych
"""

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)) # losowa permutacja indeksów - miesza dane losowo
    split = int(len(X) * (1 - test_size)) # miejsce gdzie dane mają być podzielone na 2 części
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]] # zwraca 4 tablice X_train, X_test, y_train, y_test

"""
standard_scale() - wykonuje standaryzację cech (kolumn) metodą Z‑score, czyli przekształca dane tak, aby każda cecha miała:
	•	średnią (mean) = 0
	•	odchylenie standardowe (std) = 1
Wykonujemy ją tylko na zbiorze uczącym tak aby model nie uczył się na informacjach, których jeszcze nie zna. 
Standaryzacja jest potrzebna aby cechy były porównywalne (cechy o większych wartościach mogłyby dominować nad innymi).
"""

def standard_scale(X_train: np.ndarray, X_test: np.ndarray):
        mean = X_train.mean(axis=0) #obliczana jest średnia wartość dla każdej cechy (kolumny) osobno
        std = X_train.std(axis=0) + 1e-8 #obliczane jest odchylenie standardowe również dla każdej cechy osobno, ale dodajemy małą stałą, by uniknąć dzielenia przez 0 (gdy cecha ma stałą wartość)
        return (X_train - mean) / std, (X_test - mean) / std #liczony jest Z-score

            ### FUNKCJE AKTYWACJI ###
"""
ReLU (Rectified Linear Unit) - zwraca x, gdy x > 0, w przeciwnym razie 0.
Wprowadza nieliniowość, co umożliwia sieci rozpoznawać złożone wzorce. Jest prosta i szybka do obliczenia.
Pomaga unikać problemu zanikania gradientów, w przeciwieństwie np. do sigmoidy. Działa dobrze w praktyce, szczególnie w głębszych sieciach.
Wadą jest problem martwych neuronów, gdy x jest mniejsze od 0 gradient = 0.

Problem zanikania gradientów pojawia się właśnie w głębokich sieciach neuronowych (z wieloma warstwami), podczas propagacji wstecznej, gdyż pochodne są mnożone wzdłuż warstw. 
Im głębiej w sieci -> tym mniejsze gradienty -> gradienty zanikają -> wagi nie uczą się.
Efektem jest, że wczesne warstwy uczą się bardzo wolno, albo w ogóle, a wtedy sieć się nie poprawia.
"""

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

"""
Pochodna ReLU - wynosi 1 tam, gdzie x > 0, w przeciwnym razie 0.
Potrzebna podczas wstecznej propagacji (backpropagation - sieć uczy się na podstawie błędu loss, czyli różnicy między predykcją a etykietą
i zmienia swoje wagi aby poprawić wynik) w procesie treningu. Pochodna mówi, w którą stronę i jak mocno zmieniać wagę, żeby zmniejszyć błąd sieci.
Jeśli pochodna ~ 1, to zmiana jest duża i nauka jest szybka, jednak gdy ~0 uczenie staje i mamy do czynienia z zanikaniem gradientów. 
Dzięki pochodnej ReLU wynoszącej 1 gradienty nie maleją.
"""

def relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)

"""
Tanh (skalowana hiperboliczna tangens) -  przyjmuje wartości od (-1,1).
Symetryczna wokół zera – pomaga, gdy dane wejściowe są również znormalizowane do zera (czyli po Z-score standaryzacji).
Jednak tanh cierpi na problem zanikających gradientów, w szczególności dla dużych |x|. 
Dobrze sprawdzi się w sieciach z małą liczbą warstw.
"""

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

"""
Sigmoid - mapuje liczby rzeczywiste na zakres (0, 1)
Idealna na warstwę wyjściową binarnej klasyfikacji, którą my tu zastosowaliśmy.
Charakteryzuje się „spłaszczonymi” końcami, co przy dużych |x| może powodować zanik gradientu (dlatego w ukrytych warstwach częściej używa się ReLU/tanh).
"""

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

# Słownik, który pozwala nam łatwy wybór aktywacji po nazwie
ACTIVATIONS = {
    "relu": (relu, relu_deriv),
    "tanh": (tanh, tanh_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
}

            ### DEFINICJA WARSTWY GĘSTEJ ###
class DenseLayer:

    """Inicjalizacja - budowa warstwy (wymyślenie wag, ustawianie biasów oraz funkcji aktywacji).
    input_size: ile neuronów ma wejście
    output_size: ile neuronów będzie miała ta warstwa
    activation: jaką funkcje aktywacji wybierzemy
    """

    def __init__(self, input_size: int, output_size: int, activation: str):
        limit = 1 / math.sqrt(input_size) # limit dla losowania wag, żeby zapobiegać zanikaniu gradientów
        rng = np.random.default_rng()
        self.W = rng.uniform(-limit, limit, (input_size, output_size)) # wagi W to macierz input_size * output_size, inicjalizowana z zakresu [-limit, imit]
        self.b = np.zeros((1, output_size)) # biasy "b" wektory długości output_size
        self.act, self.act_grad = ACTIVATIONS[activation] # pobranie funkcji aktywacji oraz jej pochodnej ze słownika ACTIVATIONS

        # Zmienna z oraz a_prev są zapamiętywane (cacheowane)
        self.z: np.ndarray | None = None
        self.a_prev: np.ndarray | None = None

    """
    Forward - propagacja w przód.
    a_prev: dane wejściowe do warstwy (np. dane treningowe)
    """

    def forward(self, a_prev: np.ndarray):
        self.a_prev = a_prev               # zapamiętujemy wejście, bo przyda się w backpropagation
        self.z = a_prev @ self.W + self.b  # mnożenie macierzy a_prev * W + bias
        return self.act(self.z)            # zwracamy wynik funkcji aktywacji - wyjście wartstwy

    """
    Backward - propagacja wstecz, poprawianie błędów - uczenie się. Sprawdzamy jak bardzo
    każde wejście i każda waga przyczyniła się do błedu. Liczymy jak zmienić wagi i biasy, 
    żeby następnym razem było lepiej
    dA: gradient błędu względem aktywacji wyjściowej tej warstwy
    """

    def backward(self, dA: np.ndarray):
        dz = dA * self.act_grad(self.z)    # łańcuch pochodnych: dL/dz = dL/dA * dA/dz, jak bardzo wynik z tej warstwy był winny błędu
        dW = self.a_prev.T @ dz / len(dz) # gradient wag, mnożenie transponowanego wejścia przez uśrednione po długości dz
        db = dz.mean(axis=0, keepdims=True) # gradient biasu, to średnia z dz
        dA_prev = dz @ self.W.T # gradient względem wejścia a_prev, przekazywany do poprzedniej warstwy (jeśli jest)
        return dA_prev, dW, db

            ### STRUKTURA CAŁEJ SIECI ###
class MLP:

    """
    To jest prosty model sieci neuronowej (MLP - Multilayer Perceptron) do binarnej klasyfikacji (kilka warstw Dense + 1 neuron wyjściowy).
    Oznacza to, że uczy się przewidywać, czy coś należy do jednej z dwóch klas, w tym przypadku “0” czy “1”.
    input_dim: liczba cech wejściowych
	num_layers: ile warstw ukrytych (czyli „neuronów pośrednich”) będzie między wejściem a wyjściem.
	num_neurons: ile neuronów ma każda ukryta warstwa.
	activation: funkcja aktywacji
	learning_rate: tempo uczenia (krok aktualizacji wagi)
    """

    def __init__(self, input_dim: int, num_layers: int, num_neurons: int,
                 activation: str, learning_rate: float):
        self.layers: list[DenseLayer] = [] # tworzymy pustą listę warstw ukrytych
        dim_prev = input_dim # zmienna pomocnicza - aktualna liczba wejść do danej warstwy (na początku równa input_dim)

        # Tworzymy warstwę typu DenseLayer o odpowiednich wymiarach i aktywacji.
        # Aktualizujemy dim_prev, ponieważ następna warstwa dostanie na wejściu num_neurons neuronów z poprzedniej.
        for _ in range(num_layers):
            self.layers.append(DenseLayer(dim_prev, num_neurons, activation))
            dim_prev = num_neurons
        # Pojedynczy neuron wyjściowy + sigmoid
        limit = 1 / math.sqrt(dim_prev) # limit dla inicjalizacji wag wyjściowych (zapobiega zbyt dużym wartościom)
        rng = np.random.default_rng()
        self.out_W = rng.uniform(-limit, limit, (dim_prev, 1)) # wagi warstwy wyjściowej z zakresu (-limit, limit), rozmiar: (liczba neuronów ostatniej warstwy, 1)
        self.out_b = np.zeros((1, 1)) # bias warstwy wyjściowej jako zero
        self.lr = learning_rate

    """
    Funkcja wykonująca przejście sygnału przez całą sieć dla danych wejściowych X.
    """

    def forward(self, X: np.ndarray):
        a = X # aktualna aktywacja, początkowa równa wejściu X

        # Dla każdej warstwy ukrytej wykonujemy propagację w przód, aktualizując a
        for layer in self.layers:
            a = layer.forward(a)
        z = a @ self.out_W + self.out_b # liczona jest kombinacja aktywacji ostatniej warstwy z wagami wyjściowymi + bias
        y_hat = sigmoid(z) # sigmoida daje prawdopodobieństwo klasy "1"
        self._cache = (a, z) # zachowujemy w cache, by policzyć backward
        return y_hat # zwracamy przewidziane prawdopodobieństwo klasy "1"

    """
    Loss function - jej celem jest zmierzenie, jak bardzo przewidywania sieci różnią się od prawdziwych odpowiedzi.
    """

    def compute_loss(self, y_hat: np.ndarray, y_true: np.ndarray):
        eps = 1e-8                        # zapobiega log(0) oraz dzieleniu przez 0

        # Obliczana jest średnia wartość binary cross-entropy - standardowa funkcja kosztu dla klasyfikacji binarnej
        return -np.mean(y_true * np.log(y_hat + eps) + (1 - y_true) * np.log(1 - y_hat + eps))

    """
    Funkcja propagacji wstecznej wywoływana w celu obliczenia gradientów i aktualizacji wag.
    """

    def backward(self, y_hat: np.ndarray, y_true: np.ndarray):
        a_last, _ = self._cache # pobranie z cache aktywacji przed warstwą wyjściową
        dz_last = (y_hat - y_true) / len(y_true) # pochodna funkcji kosztu względem "z" dla warstwy wyjściowej
        dW_out = a_last.T @ dz_last # gradient wag wyjściowych
        db_out = dz_last.mean(axis=0, keepdims=True) # gradient biasu wag wyjściowych
        dA_prev = dz_last @ self.out_W.T # gradient aktywacji ostatniej ukrytej warstwy
        grads = [] # lista do przechowywania gradientów każdej ukrytej warstwy

        # Cofamy się przez każdą ukrytą warstwę, obliczamy gradienty, zapisujemy je
        for layer in reversed(self.layers):
            dA_prev, dW, db = layer.backward(dA_prev)
            grads.append((layer, dW, db))

        # Aktualizacja wag i biasów warstwy wyjściowej
        self.out_W -= self.lr * dW_out
        self.out_b -= self.lr * db_out

        # Aktualizacja wag i biasów każdej warstwy ukrytej
        for layer, dW, db in grads:
            layer.W -= self.lr * dW
            layer.b -= self.lr * db

    """
    Funkcja treningowa sieci na zbiorze X_train, y_train
    
    epochs: liczba epok, czyli ile razy model ma przejść przez cały zbiór danych treningowych
    Im więcej epok tym dłuższe uczenie. Daje to możliwość modelowi na stopniowe poprawianie
    wag na podstawie błędów. Za pierwszym razem model uczy się bardzo ogólnie. W kolejnych
    epokach poprawia swoje przewidywania, bo widzi te same dane w innej kolejności, z innymi wagami.
    Jednakże, za duża ilość epok może przeuczyć model, czyli zapamięta dane zamiast się ich nauczyć.
    
    batch_size: liczba przykładów, które model przetwarza na raz podczas jednej aktualizacji wag. Zamiast
    przetwarzać cały X_train naraz, model dzieli dane na mini-batche. Taki podział przyśpiesza i stabilizuje uczenie.
    """

    def fit(self, X_train, y_train, epochs: int, batch_size: int):
        n = len(X_train) # liczba próbek treningowych

        """
        Mieszanie kolejności próbek treningowych na początku każdej epoki uczenia,
        Gdy dane są posortowane (np. rosnąca wartość pH), sieć widzi w pierwszych krokach wyłącznie niski zakres cech. 
        To wprowadza stronniczość w początkowych aktualizacjach gradientu. Dzięki mieszaniu sieć nie może „nauczyć się” 
        sekwencji próbek, bo w każdym przejściu dostaje je w innej kolejności.
        """

        for _ in range(epochs):
            idx = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]

            # Tworzenie mini-batchów
            for start in range(0, n, batch_size):
                end = start + batch_size
                Xb, yb = X_train[start:end], y_train[start:end]

                #Dla każdego mini-batcha wykonuje się przejście w przód, przejście wstecz i aktualizacja wag
                y_hat = self.forward(Xb)
                self.backward(y_hat, yb)

    """
    Zwraca przewidywane prawdopodobieństwo klasy 1 dla podanych danych X.
    """

    def predict_proba(self, X: np.ndarray):
        return self.forward(X)

            ####  MIARY OCENY KLASYFIKATORA ###

"""
Accuracy (dokładność) - sprawdza jaki procent przewidywań był poprawny. (np. 0.95 = model zgaduje dobrze w 95% przypadków)

y_true: prawdziwe etykiety klasy (dane referencyjne, znane odpowiedzi)
y_pred_prob: przewidywane prawdopodobieństwo, wychodzi z funkcji sigmoid
thr: próg decyzyjny, jakiego minimalnego prawdopodobieństwa trzeba, by uznać, że to klasa 1
"""

def accuracy(y_true: np.ndarray, y_pred_prob: np.ndarray, thr: float = 0.5):
    y_pred = (y_pred_prob >= thr).astype(int) #progowane prawdopodobieństwa thr = 0.5 oraz zamiana True/False na 1/0
    return (y_pred == y_true).mean() #sprawdza ile predykcji było poprawnych i zwraca średnią = procent trafień

"""
Precision (precyzja) - ile z zaklasyfikowanych jako "1" było faktycznie "1", czyli jak często model ma rację. 
Oczekujemy wysokiej wartości oznaczającej mało fałszywych alarmów.

Recall (czułość) - ile z prawdziwych "1" udało się wykryć i określić jako "1".
Wysoka wartość oznacza, że sieć rzadko pomija rzeczy, które powinna wykryć

F1 - średnia harmoniczna precision i recall, przydatne gdy dane są niezbalansowane (np. 90% klas 0, 10% klas 1)
Wysokie F1 to dobry kompromis, model nie strzela na ślepo i niczego nie pomija.
"""

def precision_recall_f1(y_true: np.ndarray, y_pred_prob: np.ndarray, thr=0.5):
    y_pred = (y_pred_prob >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1

            ###  GŁÓWNA PĘTLA ###
"""
Funkcja przyjmuje zarówno dane treningowe i testowe
repeats: liczba powtórzeń treningu dla jednego zestawu hiperparametrów
"""

def run_experiments(X_train, X_test, y_train, y_test,
                    repeats: int = 5, epochs: int = 50):

    #Słownik, zawierający listy możliwych wartości dla każdego hiperparametru
    PARAM_GRID = {
        "num_layers":   [1, 2, 3, 4],
        "num_neurons":  [8, 16, 32, 64],
        "learning_rate": [0.1, 0.05, 0.01, 0.005],
        "activation":   ["tanh", "relu", "sigmoid"],
        "batch_size":   [16, 32, 64, 128],
    }

    # Stworzenie folderu na wyniki, jeśli takowy istnieje to nie tworzony jest nowy
    out_dir = Path("src/results")
    out_dir.mkdir(parents = True, exist_ok=True)

    """
    Potrzebujemy jednego punktu odniesienia (baseline), od którego będziemy odchylać tylko jeden hiperparametr naraz.
    Ustalamy „baseline” – zawsze druga wartość z listy v[1], potem zmieniamy 1 parametr naraz.
	PARAM_GRID.items() zwraca pary (nazwa_parametru, lista_wartości), gdzie k to nazwa parametru (np. 'num_layers'), a v[1] to drugi element listy wartości (indeks 1).
	Dzięki temu baseline nie leży na żadnym z krańców skali, co ułatwia późniejsze porównania.
    """

    BASELINE = {k: v[1] for k, v in PARAM_GRID.items()}

    # Pętla po wszystkich nazwach hiperparametrów (param) i odpowiadających im listach wartości (values)
    for param, values in PARAM_GRID.items():
        rows = [] # pusta lista, gdzie będą trafiać wyniki dla każdej wartości danego parametru

        # Pętla po każdej wartości v danego parametru param
        for v in values:
            cfg = BASELINE.copy(); cfg[param] = v      # stworzenie kopii BASELINE i zmieniamy tylko jeden parametr a wartość v
            tr_acc, te_acc, prec_list, recall_list, f1_list = [], [], [], [], [] # pusta lista na wyniki metryk

            # Powtarzamy trenowanie, by uśrednić wpływ losowości
            for _ in range(repeats):

                # Tworzona jest instancja modelu sieci neuronowej z odpowiednią konfiguracją hiperparametrów
                model = MLP(
                    input_dim=X_train.shape[1],
                    num_layers=cfg["num_layers"],
                    num_neurons=cfg["num_neurons"],
                    activation=cfg["activation"],
                    learning_rate=cfg["learning_rate"],
                )
                model.fit(X_train, y_train, epochs=epochs, batch_size=cfg["batch_size"]) # trenowanie modelu na zbiorze treningowym
                tr_acc.append(accuracy(y_train, model.predict_proba(X_train))) # liczona dokładność na zbiorze treningowym oraz zapisanie w liście
                te_acc.append(accuracy(y_test, model.predict_proba(X_test))) # liczona dokładność na zbiorze testowym oraz zapisanie w liście

                # Liczenie innych metryk na zbiorze testowym oraz zapisanie w listach
                p, r, f = precision_recall_f1(y_test, model.predict_proba(X_test))
                prec_list.append(p)
                recall_list.append(r)
                f1_list.append(f)

            # Tworzony jest słownik row z uśrednionymi metrykami dla danej wartości hiperparametru
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
            rows.append(row) # Dodajemy row do listy rows
            print(f"  {param} = {v:<8} | test_mean = {row['test_mean']:.3f}")
        csv_path = out_dir / f"results_{param}.csv" # tworzona jest ścieżka do pliku CSV, w którym zapiszemy wyniki dla danego hiperparametru

        # Otwierany jest plik CSV, nagłówki tworzone są na podstawie kluczy z row, wartości natomiast z rows
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Zapisano wyniki dla  parametru '{param}' do {csv_path}")

            ### ŁADOWANIE DANYCH ###
"""
Funkcja ładująca dane oraz określająca próg klasyfikacji, następnie tworzone są 
wektory X oraz y
path: ścieżka do pliku CSV
threshold: 6, próg do binarnej klasyfikacji jakości
"""

def load_dataset(path: str, threshold: int = 6):
    df = pd.read_csv(path, sep=",") #wczytanie pliku csv, przecinek jest separatorem
    df = pd.get_dummies(df, columns=["type"], drop_first=True) # przekształcamy także zmienną 'type' na 1/0

    # Tworzona jest macierz cech X, usuwana jest kolumna quality (etykieta), bo jest to zmienna docelowa,
    # .values daje macierz NumPy
    X = df.drop(columns=["quality"]).values.astype(np.float32)

    # Tworzony jest wektor etykiet y, porównanie z progiem klasyfikacji oraz zamiana na 1/0
    # .values to konwersja do tablicy NumPy i .reshape zmienia kształt to kolumnowego wektora
    y = (df["quality"] >= threshold).astype(int).values.reshape(-1, 1)
    return X, y

            ### MODEL SKLEARN ###
def run_sklearn_model(X_train, X_test, y_train, y_test, args):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    print("\n Porównanie z modelem scikit-learn (MLPClassifier)")

    # Tworzymy klasyfikator z taką samą konfiguracją jak ustawiona w main
    skl_model = MLPClassifier(
        hidden_layer_sizes=(args.hidden_units,) * args.hidden_layers, # np. 2 warstwy ukryte, każda z 32 neuronami
        activation=args.activation,
        learning_rate_init=args.lr,
        max_iter=args.epochs,
        batch_size=args.batch,
        solver="sgd",
        random_state=42
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        skl_model.fit(X_train, y_train.ravel())
    y_pred = skl_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy  : {acc:.3f}")
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec:.3f}")
    print(f"F1-score  : {f1:.3f}")

            ### PARSER ARGUMENTÓW ###
"""
Funkcja parse_args, która zwraca obiekt z argumentami wiersza poleceń args.
"""

def parse_args():
    p = argparse.ArgumentParser() # tworzymy parser argumentów
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

            ### MAIN ###
def main(args):
    #args = parse_args() # pobranie argumentu z wiersza poleceń i przypisanie do zmiennej args
    print("\nWczytywanie danych")
    X, y = load_dataset(args.file, threshold=args.threshold) # wczytanie danych X i etykiet y, stosując próg jakości treshold
    print(f"Dane: {X.shape[0]} próbek, {X.shape[1]} cech")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size) # dzielenie danych zgodnie z parametrem --test_size
    X_train, X_test = standard_scale(X_train, X_test) # standaryzacja danych
    print(f"Podział danych: Treningowe: {len(X_train)} | Testowe: {len(X_test)}")

    # jeśli podano flagę experiments to wykonujemy eksperymenty
    if args.experiments:
        start = perf_counter() # czas rozpoczęcia
        run_experiments(X_train, X_test, y_train, y_test, repeats=5, epochs=args.epochs)
        print(f"Cały proces eksperymentów zajął {perf_counter()-start:.1f} sekund")

    # jeśli nie podano tej flagi, trenujemy pojedynczy model
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
        dur = perf_counter() - start # czas trwania treningu
        train_acc = accuracy(y_train, model.predict_proba(X_train)) # obliczamy dokładność dla zbioru treningowego
        test_acc = accuracy(y_test, model.predict_proba(X_test)) # obliczamy dokładność dla zbioru testowego
        prec, rec, f1 = precision_recall_f1(y_test, model.predict_proba(X_test))# liczymy inne metryki dla zbioru treningowego i testowego
        print(f"\nCzas uczenia: {dur:.1f} s | train_acc = {train_acc:.3f} | test_acc={test_acc:.3f} | prec={prec:.3f} | rec={rec:.3f} | F1={f1:.3f}")
        run_sklearn_model(X_train, X_test, y_train, y_test, args)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        class Args:
            file = "/Users/filipekmac/Documents/GitHub/classification-model-from-scratch/src/data/wine-quality-red.csv"
            threshold = 6
            test_size = 0.2
            hidden_layers = 3
            hidden_units = 32
            activation = "tanh"
            lr = 0.1
            batch = 16
            epochs = 50
            experiments = False
        args = Args()
    main(args)