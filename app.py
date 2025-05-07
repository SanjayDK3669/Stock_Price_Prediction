import numpy as np
import pandas as pd
#import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
#from datetime import date, timedelta

# date.today()


model = load_model("Stock_Prediction_model.keras")
st.header("Stock Price Predictor")
stock = st.text_input("Enter stack Symbol", "GOOG")

# start = '2014-01-01'
# end = date.today()  + timedelta(days=1)

# data = yf.download(stock, start, end)

from alpha_vantage.timeseries import TimeSeries
import pandas

api_key = 'Y3CYFY9T7LY6LY6S'

ts = TimeSeries(key = api_key, output_format = 'pandas')

data, meta_data = ts.get_daily(symbol = stock, outputsize = 'full')

data.rename(columns= {'1. open': 'Open', '2. high': 'high', '3. low': 'low', '4. close': 'Close', '5. volume': 'Volume'}, inplace= True)

#data = data.sort_index( ascending=True) 

st.subheader("Stock Data")
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.head(100)
data_test = pd.concat([past_100_days, data_test], ignore_index = True)
data_test_scaler = scaler.fit_transform(data_test)

st.subheader("Price v/s Moving Average 50")
ma_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 8))
plt.plot(ma_50, 'r', label = "MA50")
plt.plot(data.Close, 'g', label = "Original")
plt.show()
plt.legend()
st.pyplot(fig1)

st.subheader("Price v/s Moving Average 50  v/s Moving Average 100")
ma_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 8))
plt.plot(ma_50, 'r', label = "MA50")
plt.plot(ma_100, 'b', label = "MA100")
plt.plot(data.Close, 'g', label = "Original")
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader("Price v/s Moving Average 100  v/s Moving Average 200")
ma_200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 8))
plt.plot(ma_100, 'r', label = "MA100")
plt.plot(ma_200, 'b', label = "MA200")
plt.plot(data.Close, 'g', label = "Original")
plt.show()
plt.legend()
st.pyplot(fig3)



x = []
y = []

for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100: i])
    y.append(data_test_scaler[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale 
y = y * scale

# st.subheader("Original Price v/s Predicted Price")
# fig4 = plt.figure(figsize=(8, 8))
# plt.plot(predict, 'r', label = "Predicted Price")
# plt.plot(y, 'b', label = "Original Prince")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.show()
# plt.legend()
# st.pyplot(fig4)

# Use only the first 100 days' closing prices for future prediction
last_100_days = data.Close[:100].values.reshape(-1, 1)
last_100_scaled = scaler.transform(last_100_days)

future_preds = []
input_seq = last_100_scaled.reshape(1, 100, 1)

for _ in range(2):  # Predict next 2 days
    next_scaled = model.predict(input_seq)[0][0]
    future_preds.append(next_scaled)
    
    # Append predicted value to input and slide the window
    input_seq = np.append(input_seq[:, 1:, :], [[[next_scaled]]], axis=1)

# Inverse scale the predictions to get actual price
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

# Display predictions
st.subheader("Next 2 Days Predicted Closing Prices")
for i, price in enumerate(future_preds, start=1):
    st.write(f"Day {i}: ${price:.2f}")