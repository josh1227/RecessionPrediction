import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import os

df = pd.read_csv("USTREASURY-YIELD.csv", index_col="Date", parse_dates=True)
dfcurve = df["10 YR"] - df["3 MO"]
curve = np.array(dfcurve)
curve = np.flip(curve)
curve = curve.reshape(len(curve), 1)

recession = df["Recession"]
print(recession.head())
recession = np.array(recession)
recession = np.flip(recession)
recession = recession.reshape(len(recession), 1)

compare = pd.read_csv("^DJI.csv", index_col="Date", parse_dates=True)
compare = compare["Open"]

data = np.hstack([curve, recession])
training_data = data[:int(data.shape[0]*0.5)]
testing_data = data[int(data.shape[0]*0.5)-50:]

sc = MinMaxScaler(feature_range=(0, 1))
training_data = sc.fit_transform(training_data)
testing_data = sc.transform(testing_data)

def create_dataset(arr1):
    x = []
    y = []
    for i in range(50, len(arr1)):
        x.append(arr1[i-50:i, 0])
        y.append(arr1[i, 1])
    x = np.array(x)
    y = np.array(y)
    return x, y


x_train, y_train = create_dataset(training_data)
x_test, y_test = create_dataset(testing_data)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_train.shape)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(loss="mean_squared_error", optimizer="adam")

if not os.path.exists("curvePredict1.h5"):
    model.fit(x_train, y_train, epochs=50, batch_size=43)
    model.save("curvePredict1.h5")

model = load_model("curvePredict1.h5")
predictions = model.predict(x_test)
predictions = np.round(predictions)
for i in range(len(predictions)):
    if predictions[i] > 1:
        predictions[i] = 1

correct = 0
incorrect = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        correct = correct + 1
    else:
        incorrect = incorrect + 1
acc = correct/len(predictions)
print(acc)


def numRecessions(arr1):
    begin = []
    finish = []
    k = 0
    while k < len(arr1)-1:
        start = k
        while k < len(arr1)-1 and arr1[k] == arr1[k+1]:
            k = k + 1
        end = k
        if arr1[k] !=0:
            begin.append(start)
            finish.append(end)
        k = k + 1
    return begin, finish


predicted_range_begin, predicted_range_finish = numRecessions(predictions)
actual_range_begin, actual_range_finish = numRecessions(recession)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfcurve, color="black", label="Yield Curve")
ax.set_title("Predicting a Historic Recession based on Past Yield Curve Values")
ax.set_xlabel("Date")
ax.set_ylabel("Yield Curve Value")
fig.autofmt_xdate()
plt.legend()

figure, axis = plt.subplots(figsize=(10, 5))
axis.plot(compare, color="black", label="Dow Jones Industrial Average")
axis.set_title("Recession Prediction Compared to the Dow Jones Industrial Average")
axis.set_xlabel("Date")
axis.set_ylabel("Price")
figure.autofmt_xdate()
plt.legend()

orig = datetime(1990, 1, 31)

for i in range(len(predicted_range_begin)):
    start = orig + timedelta(days=(365/12)*(len(y_train)+50+predicted_range_begin[i]))
    end = orig + timedelta(days=(365/12)*(len(y_train)+50+predicted_range_finish[i]))
    print("Predicted Recession: ", start, "to", end)
    ax.axvspan(start, end, color="black", alpha=0.2, label="Predicted Recession")
    axis.axvspan(start, end, color="black", alpha=0.2, label="Predicted Recession")
for i in range(len(actual_range_begin)):
    start = orig + timedelta(days=(365/12)*(actual_range_begin[i]))
    end = orig + timedelta(days=(365/12)*(actual_range_finish[i]))
    print("Actual Recession: ", start, "to", end)
    if i == 0:
        ax.axvspan(start,  end, color="black", alpha=0.5, label="Historic Recession")
        axis.axvspan(start, end, color="black", alpha=0.5, label="Historic Recession")
    else:
        ax.axvspan(start, end, color="black", alpha=0.5)
        axis.axvspan(start, end, color="black", alpha=0.5)

ax.legend()
axis.legend()
plt.show()