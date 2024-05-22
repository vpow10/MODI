import numpy as np
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from dane_dynamiczne import load_dynamic_data


def create_data(N: int):
    train_data, test_data = load_dynamic_data()
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # Stworzenie wektora danych wejściowych i wyjściowych
    input_data = []
    output_data = []
    for i in range(N-1, len(train_data[:, 0])-1):
        row = []
        for j in range(1, N):
            row.append(train_data[i - j, 0])
            row.append(train_data[i - j, 1])
        input_data.append(row)
        output_data.append(train_data[i, 1])
    return np.array(input_data), np.array(output_data)

print(create_data(3))

def neural_network(N: int, k: int, recursive: bool, visualize: bool = True):
    # N - rząd dynamiki, k - ilość neuronów w warstwie ukrytej,
    # recursive - czy model ma być rekurencyjny, visualize - czy model ma być wizualizowany
    train_data, test_data = load_dynamic_data()
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    input_data = create_input_data(N)
    # Stworzenie modelu
    model = Sequential()


# # Example provided data (replace with actual provided data)
# train_data, test_data = load_dynamic_data()
# train_data = np.array(train_data)
# test_data = np.array(test_data)

# # Normalize the y-values (second column)
# scaler = MinMaxScaler(feature_range=(0, 1))
# train_data[:, 1] = scaler.fit_transform(train_data[:, 1].reshape(-1, 1)).reshape(-1)
# test_data[:, 1] = scaler.transform(test_data[:, 1].reshape(-1, 1)).reshape(-1)

# # Prepare the data for LSTM
# def create_dataset(data, time_step=1):
#     X, y = [], []
#     for i in range(len(data)-time_step-1):
#         a = data[i:(i+time_step), 1]
#         X.append(a)
#         y.append(data[i + time_step, 1])
#     return np.array(X), np.array(y)

# time_step = 9  # Adjust based on your requirements
# X_train, y_train = create_dataset(train_data, time_step)
# X_test, y_test = create_dataset(test_data, time_step)

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Build the model
# model = Sequential()
# model.add(LSTM(100, input_shape=(time_step, 1), activation='tanh'))
# model.add(Dense(1, activation='tanh'))

# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the Model
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# # Recursive prediction function
# def recursive_predict(model, data, time_step, num_predictions):
#     predictions = []
#     current_input = data[-1]  # Start with the last input sequence

#     for _ in range(num_predictions):
#         current_input = current_input.reshape(1, time_step, 1)
#         next_pred = model.predict(current_input)
#         predictions.append(next_pred[0, 0])

#         # Update the current input with the new prediction
#         current_input = np.append(current_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

#     return np.array(predictions)

# # Make recursive predictions
# num_predictions = len(y_test)
# predictions = recursive_predict(model, X_test, time_step, num_predictions)

# # Inverse transform the predictions
# predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

# # Plotting the results
# plt.plot(range(len(y_test)), scaler.inverse_transform(y_test.reshape(-1, 1)), label='True Data')
# plt.plot(range(len(predictions)), predictions, label='Predicted Data')
# plt.legend()
# plt.show()
