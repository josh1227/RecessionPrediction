import pandas as pd
from numpy import array
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import os

df = pd.read_csv("USTREASURY-YIELD.csv")
curve = df["10 YR"] - df["3 MO"]
training_data = curve[:263]
testing_data = curve[263:]
print(training_data.shape)
print(testing_data.shape)


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


n_steps_in, n_steps_out = 95, 50

X, y = split_sequence(training_data, n_steps_in, n_steps_out)
print(X.shape, y.shape)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

model = Sequential()
model.add(LSTM(units=95, return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=95, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=95, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=95))
model.add(Dropout(0.2))
model.add(Dense(units=n_steps_out))

model.compile(optimizer='adam', loss='mse')

if not os.path.exists("future2.h5"):
    model.fit(X, y, epochs=50, batch_size=17)
    model.save("future2.h5")

model = load_model("future2.h5")

x_input = array(testing_data)
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input)
predictions = yhat[0]

fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(predictions, color="red", label="future value")
plt.legend()
plt.show()

print(yhat)