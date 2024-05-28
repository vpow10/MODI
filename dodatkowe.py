import numpy as np
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
import matplotlib.pyplot as plt
from dane_dynamiczne import load_dynamic_data


def create_data(N: int, data_type: bool, recursive: bool):
    train_data, test_data = load_dynamic_data()
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    data = train_data if data_type else test_data
    # Stworzenie wektorów danych wejściowych i wyjściowych
    input_data = []
    output_data = []
    for i in range(N-1, len(data[:, 0])):
        row = []
        for j in range(1, N):
            row.append(data[i - j, 0])
            row.append(data[i - j, 1])
        input_data.append(row)
        output_data.append(data[i, 1])
    input_data = np.array(input_data)
    if recursive:
        input_data = input_data.reshape((input_data.shape[0], N-1, 2))
    return input_data, np.array(output_data)

def neural_network(N: int, k: int, recursive: bool, visualize: bool = True):
    # N - rząd dynamiki, k - ilość neuronów w warstwie ukrytej,
    # recursive - czy model ma być rekurencyjny, visualize - czy model ma być wizualizowany

    # Uzyskanie danych
    train_data, test_data = load_dynamic_data()
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    input_data, output_data = create_data(N, True, recursive)

    # Stworzenie modelu
    model = Sequential()
    if recursive:
        model.add(LSTM(k, input_shape=(N-1, 2), activation='leaky_relu'))
    else:
        model.add(Dense(k, input_dim=2 * (N - 1), activation='leaky_relu'))
    model.add(Dense(1, activation='leaky_relu'))

    # Uczenie modelu
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(input_data, output_data, epochs=100)

    # Stworzenie wektorów z danych testujących i predykcje modelu
    predictions_train = model.predict(input_data)
    input_data_test, output_data_test = create_data(N, False, recursive)
    predictions_test = model.predict(input_data_test)

    # Sprowadzenie predykcji do jednego wymiaru
    predictions_train = predictions_train.flatten()
    predictions_test = predictions_test.flatten()

    # Obliczenie błędu średniokwadratowego
    tse_train = round(np.sum((output_data - predictions_train.flatten())**2), 3)
    tse_test = round(np.sum((output_data_test - predictions_test.flatten())**2), 3)

    if visualize:
        recursive_str = 'z rekurencją' if recursive else 'bez rekurencji'
        plt.figure(1)
        plt.plot(range(len(output_data)), output_data, label='Zbiór uczący')
        plt.plot(range(len(output_data)), predictions_train, label='Wyjście modelu')
        plt.title(f'Model {recursive_str}, dane uczące')
        plt.xlabel('k')
        plt.ylabel('y')
        plt.legend()
        plt.figure(2)
        plt.plot(range(len(output_data_test)), output_data_test, label='Zbiór testujący')
        plt.plot(range(len(output_data_test)), predictions_test, label='Wyjście modelu')
        plt.title(f'Model {recursive_str}, dane testujące')
        plt.xlabel('k')
        plt.ylabel('y')
        plt.legend()
        print(f'Błąd średniokwadratowy dla zbioru uczącego: {tse_train}')
        print(f'Błąd średniokwadratowy dla zbioru testującego: {tse_test}')
        plt.show()
    return tse_train, tse_test

if __name__ == "__main__":
    # Zbyt mała (1) ilość neuronów w warstwie ukrytej
    neural_network(8, 1, recursive=False)
    neural_network(8, 1, recursive=True)
    # Optymalna ilość neuronów w warstwie ukrytej
    neural_network(8, 20, recursive=False)
    neural_network(8, 20, recursive=True)