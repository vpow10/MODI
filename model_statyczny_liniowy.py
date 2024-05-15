import numpy as np
import matplotlib.pyplot as plt
from dane_statyczne import split_static_data, load_static_data


# Model statyczny liniowy

def linear_static_model():
    # wczytanie danych
    data = load_static_data()
    data = np.array(data)
    train, test = split_static_data()
    # trenowanie modelu
    Y = train[:, 1]
    M = [np.ones(len(train)), train[:, 0]]
    M = np.array(M).T
    w = np.linalg.lstsq(M, Y, rcond=None)[0]
    # wizualizacja modelu statycznego
    plt.figure(1)
    plt.scatter(data[:, 0], w[0] + w[1] * data[:, 0], c='b', linewidths=0.5, edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model statyczny liniowy")
    # wizualizacja modelu na tle danych uczących
    plt.figure(2)
    plt.scatter(train[:, 0], train[:, 1], c='r', linewidths=0.5, edgecolors='black')
    plt.scatter(train[:, 0], w[0] + w[1] * train[:, 0], c='b', linewidths=0.5, edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model statyczny liniowy na tle danych uczących")
    # wizualizacja modelu na tle danych testowych
    plt.figure(3)
    plt.scatter(test[:, 0], test[:, 1], c='r', linewidths=0.5, edgecolors='black')
    plt.scatter(test[:, 0], w[0] + w[1] * test[:, 0], c='b', linewidths=0.5, edgecolors='black')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model statyczny liniowy na tle danych testowych")
    # obliczenie błędu dla danych uczących
    Y_pred = w[0] + w[1] * train[:, 0]
    error_train = np.sum((Y_pred - Y) ** 2)
    print("Błąd dla danych uczących: ", round(error_train, 3))
    # obliczenie błędu dla danych testowych
    Y_pred = w[0] + w[1] * test[:, 0]
    error_test = np.sum((Y_pred - test[:, 1]) ** 2)
    print("Błąd dla danych testowych: ", round(error_test, 3))
    plt.show()
    return round(error_train, 3), round(error_test, 3)


if __name__ == "__main__":
    linear_static_model()