import numpy as np
import matplotlib.pyplot as plt


# Odczyt danych dynamicznych z pliku

def load_dynamic_data():
    train_data = []
    with open("dane/danedynucz53.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            train_data.append([float(line[0]), float(line[1])])
    test_data = []
    with open("dane/danedynwer53.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            test_data.append([float(line[0]), float(line[1])])
    return train_data, test_data

# Wizualizacja danych dynamicznych

def visualize_dynamic_data():
    train_data, test_data = load_dynamic_data()
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    k = np.linspace(0, 2000, len(train_data[:, 0]))
    plt.figure(1)
    plt.plot(k, train_data[:, 1], c='r', linewidth=1)
    plt.xlabel("k")
    plt.ylabel("y")
    plt.title("Dane dynamiczne - zbiór uczący")
    plt.figure(2)
    plt.plot(k, test_data[:, 1], c='b', linewidth=1)
    plt.xlabel("k")
    plt.ylabel("y")
    plt.title("Dane dynamiczne - zbiór testujący")
    plt.show()


if __name__ == "__main__":
    visualize_dynamic_data()