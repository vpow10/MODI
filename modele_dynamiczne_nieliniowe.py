import numpy as np
import matplotlib.pyplot as plt
from dane_dynamiczne import load_dynamic_data


# Model dynamiczny nieliniowy

def nonlinear_dynamic_model(N: int, K: int, recursive: bool, visualize: bool = True):
    # N - rząd dynamiki, K - stopień wielomianu, recursive - czy model ma być rekurencyjny, visualize - czy model ma być wizualizowany
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
            for k in range(1, K + 1):
                row.append(train_data[i - j, 0] ** k)
                row.append(train_data[i - j, 1] ** k)
        M.append(row)
    M = np.array(M)
    w = np.linalg.lstsq(M, Y, rcond=None)[0]

    # Obliczanie wyjścia modelu

    if recursive:

        # Model dynamiczny nieliniowy z rekurencją

        ymod_train = []
        # Inicjalizacja modelu
        ymod_train[:N] = train_data[:N, 1]
        # Wyjście modelu dla danych uczących

        for i in range(N-1, len(train_data[:, 0])-1):
            # Stworzenie wiersza macierzy na podobieństwo macierzy M, ale z rekurencją, a więc z wyjściem modelu
            row = []
            for j in range(N):
                for k in range(1, K + 1):
                    row.append(train_data[i - j, 0] ** k)
                    row.append(ymod_train[i - j] ** k)
            # Obliczenie wyjścia modelu dla danych uczących
            temp = 0
            for i in range(len(row)):
                temp += w[i] * row[i]
            ymod_train.append(temp)

        # Wyjście modelu dla danych testujących
        ymod_test = []
        # Inicjalizacja modelu
        ymod_test[:N] = test_data[:N, 1]
        for i in range(N-1, len(test_data[:, 0])-1):
            # Stworzenie wiersza macierzy na podobieństwo macierzy M, ale z rekurencją, a więc z wyjściem modelu
            row = []
            for j in range(N):
                for k in range(1, K + 1):
                    row.append(test_data[i - j, 0] ** k)
                    row.append(ymod_test[i - j] ** k)
            # Obliczenie wyjścia modelu dla danych testujących
            temp = 0
            for i in range(len(row)):
                temp += w[i] * row[i]
            ymod_test.append(temp)
    else:

        # Model dynamiczny nieliniowy bez rekurencji

        # wyjście modelu dla danych uczących
        ymod_train = []
        for row in M:
            temp = 0
            for i in range(len(row)):
                temp += w[i] * row[i]
            ymod_train.append(temp)
        # wyjście modelu dla danych testujących
        ymod_test = []
        for i in range(N-1, len(test_data[:, 0])-1):
            # Stworzenie wiersza macierzy na podobieństwo macierzy M, ale dla danych testujących
            row = []
            for j in range(N):
                for k in range(1, K + 1):
                    row.append(test_data[i - j, 0] ** k)
                    row.append(test_data[i - j, 1] ** k)
            # Obliczenie wyjścia modelu dla danych testujących
            temp = 0
            for i in range(len(row)):
                temp += w[i] * row[i]
            ymod_test.append(temp)

    # Wizualizacja modelów

    if visualize:
        if recursive:
            # Wizualizacja modelu dynamicznego na tle danych uczących
            x = np.linspace(0, 1, len(train_data[:, 0]))
            plt.figure(1)
            plt.scatter(x, train_data[:, 1], c='r', linewidths=0.5, edgecolors='black')
            plt.plot(x, ymod_train, c='b', linewidth=1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Model dynamiczny nieliniowy z rekurencją na tle danych uczących")

            # Wizualizacja modelu dynamicznego na tle danych testujących
            plt.figure(2)
            plt.scatter(x, test_data[:, 1], c='r', linewidths=0.5, edgecolors='black')
            plt.plot(x, ymod_test, c='b', linewidth=1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Model dynamiczny nieliniowy z rekurencją na tle danych testujących")
        else:
            # Wizualizacja modelu dynamicznego na tle danych uczących
            x = np.linspace(0, 1, len(Y))
            plt.figure(1)
            plt.scatter(x, Y, c='r', linewidths=0.5, edgecolors='black')
            plt.plot(x, ymod_train, c='b', linewidth=1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Model dynamiczny nieliniowy bez rekurencji na tle danych uczących")

            # Wizualizacja modelu dynamicznego na tle danych testujących
            plt.figure(2)
            plt.scatter(x, test_data[N:, 1], c='r', linewidths=0.5, edgecolors='black')
            plt.plot(x, ymod_test, c='b', linewidth=1.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Model dynamiczny nieliniowy bez rekurencji na tle danych testujących")


    # Oblliczenie błędu dla danych uczących
    error_train = np.sum((ymod_train - Y) ** 2) if not recursive else np.sum((ymod_train - train_data[:, 1]) ** 2)
    # Obliczenie błędu dla danych testujących
    error_test = np.sum((ymod_test - test_data[N:, 1]) ** 2) if not recursive else np.sum((ymod_test - test_data[:, 1]) ** 2)
    if visualize:
        print("Błąd dla danych uczących: ", round(error_train, 3))
        print("Błąd dla danych testowych: ", round(error_test, 3))
        plt.show()
    return w, round(error_train, 2), round(error_test, 2)

def test_multiple_params(max_dynamic_degree: int, max_polynomial_degree: int, recursive: bool):
    all_errors_train = {i: [] for i in range(1, max_dynamic_degree + 1)}
    all_errors_test = {i: [] for i in range(1, max_dynamic_degree + 1)}
    for i in range(1, max_dynamic_degree + 1):
        errors_train = []
        errors_test = []
        for j in range(1, max_polynomial_degree + 1):
            _, error_train, error_test = nonlinear_dynamic_model(i, j, recursive, False)
            errors_train.append(error_train)
            errors_test.append(error_test)
        all_errors_train[i] = errors_train
        all_errors_test[i] = errors_test
    for i, errors_test in all_errors_test.items():
        print(f"Stopień dynamiki: {i}, najmniejszy błąd dla danych testowych: {min(errors_test)} dla wielomianu stopnia {errors_test.index(min(errors_test)) + 1}")
    return all_errors_train, all_errors_test

def make_static_characteristic(N: int, K: int, recursive: bool):
    _, test_data = load_dynamic_data()
    test_data = np.array(test_data)
    u = np.linspace(-1, 1, 2000)
    w, _, _ = nonlinear_dynamic_model(N, K, recursive, False)
    y = []
    # Inicjalizacja modelu
    y[:N] = test_data[:N, 1]
    for i in range(N-1, len(u)-1):
        # Stworzenie wiersza macierzy na podobieństwo macierzy M, ale z rekurencją, a więc z wyjściem modelu
        row = []
        for j in range(N):
            for k in range(1, K + 1):
                row.append(u[i - j] ** k)
                row.append(y[i - j] ** k)
        # Obliczenie wyjścia modelu dla danych testujących
        temp = 0
        for i in range(len(row)):
            temp += w[i] * row[i]
        y.append(temp)
    plt.figure(1)
    plt.plot(u[:45], y[:45], c='r', linewidth=1.5)
    plt.plot(u[45:], y[45:], c='b', linewidth=1.5)
    plt.xlabel("u")
    plt.ylabel("y")
    plt.title("Charakterystyka statyczna na podstawie najlepszego modelu dynamicznego")
    plt.show()


if __name__=="__main__":
    # print(test_multiple_params(8, 6, False))
    # print(test_multiple_params(8, 6, True))
    # Charakterystyka statyczna na podstawie najlepszego modelu dynamicznego
    # make_static_characteristic(8, 4, True)
    nonlinear_dynamic_model(3, 1, True)