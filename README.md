# classification-model-from-scratch/main.py

Użycie tego skryptu jest możliwe na 4 sposoby:

1. W przypadku chęci iterowania przez różne hiperparametry ustawione w skrypcie wywołujemy skrypt w ten sposób, wtedy powstaną końcowe pliki .csv:

python3 src/main.py --file src/data/wine-quality-red.csv --experiments

2. Wywołanie pojedynczego modelu z możliwością modyfikowania parametrów w komendzie:

python3 src/main.py --file src/data/wine-quality-red.csv --threshold 6 --test_size 0.2 --hidden_layers 2 --hidden_units 32 --activation relu --lr 0.01 --batch 32 --epochs 50

3. Wywołanie pojedynczego modelu z domyślenie ustawionymi wartościami w skrypcie:

python3 src/main.py --file src/data/wine-quality-red.csv

4. Uruchomienie skryptu za pomocą RUN w interpreterze, możliwość zmiany parametrów w ostanim bloku if __name__ == "__main__"

W każdym przypadku wywołania pojedynczego modelu będą wyświetlane również metryki z zaimplentowanej funkcji z pakietu sklearn dla porównania
poprawności modelu pisanego od zera.