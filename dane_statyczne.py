import numpy as np
import matplotlib.pyplot as plt


# Odczyt danych statycznych z pliku

def load_static_data():
    data = []
    with open("dane/danestat53.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            data.append([float(line[0]), float(line[1])])
    return data

# Wizualizacja danych statycznych

def visualize_static_data():
    data = load_static_data()
    data = np.array(data)
    plt.scatter(data[:, 0], data[:, 1], c='r', linewidths=0.5, edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Dane statyczne")
    plt.show()

# Podział danych statycznych na zbiór uczący i testowy (co drugi wiersz)

def split_static_data():
    data = load_static_data()
    data = np.array(data)
    train = data[::2]
    test = data[1::2]
    return train, test

# Wizualizacja zbiorów uczącego i testowego

def visualize_sets():
    train, test = split_static_data()
    plt.figure(1)
    plt.scatter(train[:, 0], train[:, 1], c='r', linewidths=0.5, edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Dane uczące")
    plt.figure(2)
    plt.scatter(test[:, 0], test[:, 1], c='b', linewidths=0.5, edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Dane testowe")
    plt.show()

if __name__ == "__main__":
    visualize_static_data()
    visualize_sets()