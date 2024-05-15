import numpy as np
import matplotlib.pyplot as plt
from dane_dynamiczne import load_dynamic_data


def linear_dynamic_model(N: int, recursive: bool, visualize: bool = True):
    # N - rząd dynamiki, recursive - czy model ma być rekurencyjny, visualize - czy model ma być wizualizowany

    # wczytanie danych
    train_data, test_data = load_dynamic_data()
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # trenowanie modelu
    Y = train_data[:, 1]
    Y = Y[N:]
    M = []
    for i in range(N-1, len(train_data[:, 0])-1):
        row = []
        for j in range(N):
            row.append(train_data[i - j, 0])
            row.append(train_data[i - j, 1])
        M.append(row)
    M = np.array(M)
    w = np.linalg.lstsq(M, Y, rcond=None)[0]
    if recursive:
        # Model dynamiczny liniowy z rekurencją

        ymod_train = []
        # Inicjalizacja modelu
        ymod_train[:N] = train_data[:N, 1]
        # wyjście modelu dla danych uczących
        for i in range(N-1, len(train_data[:, 0])-1):
            temp = 0
            for j in range(N):
                temp += w[j * 2] * train_data[i - j, 0] + w[j * 2 + 1] * ymod_train[i - j]
            ymod_train.append(temp)
        # wyjście modelu dla danych testujących
        ymod_test = []
        # Inicjalizacja modelu
        ymod_test[:N] = test_data[:N, 1]
        for i in range(N-1, len(test_data[:, 0])-1):
            temp = 0
            for j in range(N):
                temp += w[j * 2] * test_data[i - j, 0] + w[j * 2 + 1] * ymod_test[i - j]
            ymod_test.append(temp)
    else:
        # Model dynamiczny liniowy bez rekurencji

        # wyjście modelu dla danych uczących
        ymod_train = []
        for i in range(N-1, len(train_data[:, 0])-1):
            temp = 0
            for j in range(N):
                temp += w[j * 2] * train_data[i - j, 0] + w[j * 2 + 1] * train_data[i - j, 1]
            ymod_train.append(temp)
        # wyjście modelu dla danych testujących
        ymod_test = []
        for i in range(N-1, len(test_data[:, 0])-1):
            temp = 0
            for j in range(N):
                temp += w[j * 2] * test_data[i - j, 0] + w[j * 2 + 1] * test_data[i - j, 1]
            ymod_test.append(temp)
    if visualize:
        if recursive:
            # wizualizacja modelu dynamicznego na tle danych uczących
            x = np.linspace(0, 1, len(train_data[:, 0]))
            plt.figure(1)
            plt.scatter(x, train_data[:, 1], c='r', linewidths=0.5, edgecolors='black')
            plt.plot(x, ymod_train, c='b', linewidth=1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Model dynamiczny liniowy z rekurencją na tle danych uczących")
            # wizualizacja modelu dynamicznego na tle danych testujących
            plt.figure(2)
            plt.scatter(x, test_data[:, 1], c='r', linewidths=0.5, edgecolors='black')
            plt.plot(x, ymod_test, c='b', linewidth=1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Model dynamiczny liniowy z rekurencją na tle danych testujących")
        else:
            # wizualizacja modelu dynamicznego na tle danych uczących
            x = np.linspace(0, 1, len(Y))
            plt.figure(1)
            plt.scatter(x, Y, c='r', linewidths=0.5, edgecolors='black')
            plt.plot(x, ymod_train, c='b', linewidth=1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Model dynamiczny liniowy na tle danych uczących")
            # wizualizacja modelu dynamicznego na tle danych testujących
            plt.figure(2)
            plt.scatter(x, test_data[N:, 1], c='r', linewidths=0.5, edgecolors='black')
            plt.plot(x, ymod_test, c='b', linewidth=1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Model dynamiczny liniowy na tle danych testujących")
    # obliczenie błędu dla danych uczących
    error_train = np.sum((ymod_train - Y) ** 2) if not recursive else np.sum((ymod_train - train_data[:, 1]) ** 2)
    # obliczenie błędu dla danych testujących
    error_test = np.sum((ymod_test - test_data[N:, 1]) ** 2) if not recursive else np.sum((ymod_test - test_data[:, 1]) ** 2)
    if visualize:
        print("Błąd dla danych uczących: ", round(error_train, 3))
        print("Błąd dla danych testujących: ", round(error_test, 3))
        plt.show()
    return round(error_train, 3), round(error_test, 3)


if __name__ == "__main__":
    linear_dynamic_model(3, True)