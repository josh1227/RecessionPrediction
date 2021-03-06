# import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import os

# reading in "US-TREASURY.csv" file
df = pd.read_csv("USTREASURY-YIELD.csv")
df_curve = df["10 YR"] - df["3 MO"]
curve = np.array(df_curve)
curve = np.flip(curve)

# predicting future yield curve vales algorithm
# initialization statement refers to 95 training data points and predicting the next 50 data points
steps_in, steps_forward = 95, 50

# splitting into training data and testing data
training_data_future = curve[:curve.shape[0]-steps_in]
testing_data_future = curve[curve.shape[0]-steps_in:]


def split_sequence(sequence, n_steps_in, n_steps_out):
    x, y = list(), list()
    for k in range(len(sequence)):
        end_ix = k + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[k:end_ix], sequence[end_ix:out_end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


# splitting sequence into frames of 145 data points (95 in, 50 forward)
future_x, future_y = split_sequence(training_data_future, steps_in, steps_forward)
future_x = future_x.reshape((future_x.shape[0], future_x.shape[1], 1))

# defining model architecture for predicting future yield curve values
model_data = Sequential()
model_data.add(LSTM(units=steps_in, return_sequences=True, input_shape=(steps_in, 1)))
model_data.add(Dropout(0.2))
model_data.add(LSTM(units=steps_in, return_sequences=True))
model_data.add(Dropout(0.2))
model_data.add(LSTM(units=steps_in, return_sequences=True))
model_data.add(Dropout(0.2))
model_data.add(LSTM(units=steps_in))
model_data.add(Dropout(0.2))
model_data.add(Dense(steps_forward))

# compiling and training model
model_data.compile(optimizer='adam', loss='mse')
if not os.path.exists("future_value2.h5"):
    model_data.fit(future_x, future_y, epochs=50, batch_size=17)
    model_data.save("future_value2.h5")

# making predictions with model
model_data = load_model("future_value2.h5")
x_input = np.array(testing_data_future)
x_input = x_input.reshape((1, steps_in, 1))
predicted_value = model_data.predict(x_input, )
predicted_value = predicted_value[0]

# creating date indexes for predicted values
future_dates = []
first_date = pd.to_datetime(df["Date"][0])
iterate = 1
while len(future_dates) < len(predicted_value):
    future_dates.append(first_date + timedelta(days=30*iterate))
    iterate = iterate + 1
future_data = pd.DataFrame(predicted_value)
future_data = future_data.set_index(pd.to_datetime(future_dates))

# plotting predicted data points generated by the future values algorithm
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_title("Predicting Future Yield Curve Values based on Past Yield Curve Values")
ax1.set_xlabel("Date")
ax1.set_ylabel("Yield Curve Value")
ax1.plot(future_data, linestyle="--", color="black", label="Future Value")
fig1.autofmt_xdate()
ax1.legend()

# plotting historic data points with predicted data points
df_curve = pd.DataFrame(df_curve)
df_curve = df_curve.set_index(pd.to_datetime(df["Date"]))

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_title("Concatenated Input Data for Recession Prediction Algorithm")
ax2.set_xlabel("Date")
ax2.set_ylabel("Yield Curve Value")
ax2.plot(df_curve, color="black", label="Historic Data")
ax2.plot(future_data, linestyle="--", color="black", label="Future Data")
fig2.autofmt_xdate()
ax2.legend()

# concatenating historic data points predicted data points
new_curve = np.concatenate((curve, predicted_value))
new_curve = new_curve.reshape(len(new_curve), 1)

# reading in historic recession values
recession = np.array(df["Recession"])
recession = np.flip(recession)
recession = recession.reshape(len(recession), 1)

# adding zeros to recession column to match shape of concatenated yield curve array
index = recession.shape[0]
recession = recession.tolist()
while len(recession) < len(new_curve):
    recession.append(list([0]))
    index = index + 1

# combining yield curve values with recession values
recession = np.array(recession)
recession = recession.reshape(len(recession), 1)
data = np.hstack([new_curve, recession])

# recession prediction algorithm
# splitting data into training and testing sets
training_data_recession = data[:int(data.shape[0]*0.8)]
testing_data_recession = data[int(data.shape[0]*0.8)-50:]

# scaling data for efficiency
sc = MinMaxScaler(feature_range=(0, 1))
training_data = sc.fit_transform(training_data_recession)
testing_data = sc.transform(testing_data_recession)


def create_dataset(arr1):
    x = []
    y = []
    for k in range(50, len(arr1)):
        x.append(arr1[k-50:k, 0])
        y.append(arr1[k, 1])
    x = np.array(x)
    y = np.array(y)
    return x, y


# splitting data into x and y arrays
x_train, y_train = create_dataset(training_data)
x_test, y_test = create_dataset(testing_data)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# defining model architecture for predicting recession algorithm
model = Sequential()
model.add(LSTM(units=x_train.shape[1], return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=x_train.shape[1], return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=x_train.shape[1], return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=x_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# compiling and training model
model.compile(loss="mean_squared_error", optimizer="adam")
if not os.path.exists("finalPredict2.h5"):
    model.fit(x_train, y_train, epochs=50, batch_size=43)
    model.save("finalPredict2.h5")

# making predictions with model
model = load_model("finalPredict2.h5")
predictions = model.predict(x_test)
predictions = np.round(predictions)
for i in range(len(predictions)):
    if predictions[i] > 1:
        predictions[i] = 1


def num_recessions(arr1):
    begin_index = []
    finish_index = []
    k = 0
    while k < len(arr1)-1:
        start_index = k
        while k < len(arr1)-1 and arr1[k] == arr1[k+1]:
            k = k + 1
        end_index = k
        if arr1[k] != 0:
            begin_index.append(start_index)
            finish_index.append(end_index)
        k = k + 1
    return begin_index, finish_index


# determining the beginning and end of recessions from historic data and predicted data
predicted_range_begin, predicted_range_finish = num_recessions(predictions)
actual_range_begin, actual_range_finish = num_recessions(recession)

# plotting the future recession based on future yield curve values
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.set_title("Predicting a Future Recession based on Future Yield Curve Values")
ax3.set_xlabel("Date")
ax3.set_ylabel("Yield Curve Value")
ax3.plot(df_curve, color="black", label="Historic Data")
ax3.plot(future_data, linestyle="--", color="black", label="Future Data")
fig3.autofmt_xdate()

# reading in and plotting data from the Dow Jones Industrial Average to visualize historic recession impact
dow_compare = pd.read_csv("^DJI.csv", index_col="Date", parse_dates=True)
dow_compare = dow_compare["Open"]

fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.set_title("Recession Prediction Compared to the Dow Jones Industrial Average")
ax4.set_xlabel("Date")
ax4.set_ylabel("Price")
ax4.plot(dow_compare, color="black", label="Dow Jones Industrial Average")
fig4.autofmt_xdate()

# creating vertical spanning rectangles to highlight time periods of recession
last_date = pd.to_datetime(df["Date"][(df["Date"].shape[0]-1)])
for i in range(len(predicted_range_begin)):
    start_date = last_date + timedelta(days=(365/12)*(len(y_train)+50+predicted_range_begin[i]))
    end_date = last_date + timedelta(days=(365/12)*(len(y_train)+50+predicted_range_finish[i]))
    print("Dates of the Predicted Recession: ", start_date, "to", end_date)
    ax3.axvspan(start_date, end_date, color="black", alpha=0.2, label="Predicted Recession")
    ax4.axvspan(start_date, end_date, color="black", alpha=0.2, label="Predicted Recession")
for i in range(len(actual_range_begin)):
    start_date = last_date + timedelta(days=(365/12)*(actual_range_begin[i]))
    end_date = last_date + timedelta(days=(365/12)*(actual_range_finish[i]))
    print("Actual Recession: ", start_date, "to", end_date)
    if i == 0:
        ax3.axvspan(start_date, end_date, color="black", alpha=0.5, label="Historic Recession")
        ax4.axvspan(start_date, end_date, color="black", alpha=0.5, label="Historic Recession")
    else:
        ax3.axvspan(start_date, end_date, color="black", alpha=0.5)
        ax4.axvspan(start_date, end_date, color="black", alpha=0.5)

# updating figure legends and displaying all graphs
ax3.legend(loc=3)
ax4.legend()
plt.show()
