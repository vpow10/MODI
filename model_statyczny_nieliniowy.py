import numpy as np
import matplotlib.pyplot as plt
from dane_statyczne import split_static_data, load_static_data


# Model statyczny nieliniowy

def nonlinear_static_model(N: int, visualize: bool = True):
    # N - stopień wielomianu

    # wczytanie danych
    data = load_static_data()
    data = np.array(data)
    train, test = split_static_data()
    # trenowanie modelu
    Y = train[:, 1]
    M = []
    for i in range(len(train[:, 0])):
        row = []
        row.append(1)
        for j in range(1, N + 1):
            row.append(train[i, 0] ** j)
        M.append(row)
    M = np.array(M)
    w = np.linalg.lstsq(M, Y, rcond=None)[0]
    if visualize:
        # wizualizacja modelu statycznego
        plt.figure(1)
        plt.scatter(data[:, 0], np.polyval(w[::-1], data[:, 0]), c='b', linewidths=0.5, edgecolors='black')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Model statyczny nieliniowy")
        # wizualizacja modelu na tle danych uczących
        plt.figure(2)
        plt.scatter(train[:, 0], train[:, 1], c='r', linewidths=0.5, edgecolors='black')
        plt.scatter(train[:, 0], np.polyval(w[::-1], train[:, 0]), c='b', linewidths=0.5, edgecolors='black')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Model statyczny nieliniowy na tle danych uczących")
        # wizualizacja modelu na tle danych testowych
        plt.figure(3)
        plt.scatter(test[:, 0], test[:, 1], c='r', linewidths=0.5, edgecolors='black')
        plt.scatter(test[:, 0], np.polyval(w[::-1], test[:, 0]), c='b', linewidths=0.5, edgecolors='black')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Model statyczny nieliniowy na tle danych testowych")
    # obliczenie błędu dla danych uczących
    Y_pred = np.polyval(w[::-1], train[:, 0])
    error_train = np.sum((Y_pred - Y) ** 2)
    # obliczenie błędu dla danych testowych
    Y_pred = np.polyval(w[::-1], test[:, 0])
    error_test = np.sum((Y_pred - test[:, 1]) ** 2)
    if visualize:
        print(f"Błąd dla danych uczących dla wielomianu stopnia {N}: ", round(error_train, 3))
        print(f"Błąd dla danych testowych dla wielomianu stopnia {N}: ", round(error_test, 3))
        plt.show()
    return round(error_train, 3), round(error_test, 3)


def test_multiple_degrees(max_degree: int):
    errors_train = []
    errors_test = []
    for i in range(1, max_degree + 1):
        print("Stopień wielomianu: ", i)
        error_train, error_test = nonlinear_static_model(i, False)
        errors_train.append(error_train)
        errors_test.append(error_test)
    plt.figure(4)
    plt.plot(range(1, max_degree + 1), errors_train, c='r', label="Błąd dla danych uczących")
    plt.plot(range(1, max_degree + 1), errors_test, c='b', label="Błąd dla danych testowych")
    plt.xlabel("Stopień wielomianu")
    plt.ylabel("Błąd")
    plt.title("Błąd w zależności od stopnia wielomianu")
    plt.legend()
    plt.show()
    print("Najmniejszy błąd dla danych testowych:", f"{np.min(errors_test)},", "dla stopnia wielomianu:", np.argmin(errors_test) + 1)
    return errors_train, errors_test

if __name__=="__main__":
    test_multiple_degrees(20)
    # wizualizacja najlepszego wyniku
    nonlinear_static_model(4)