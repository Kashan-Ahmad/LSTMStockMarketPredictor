import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt


#model = load_model('C:\Users\kasub\OneDrive\Desktop\SimpleStockMarketPredictor\Stock Prediction Model.keras')
model = load_model(r'C:\Users\kasub\OneDrive\Desktop\SimpleStockMarketPredictor\Stock Prediction Model.keras')


st.title('📈 AI Stock Price Predictor')

stock = st.text_input("Enter Stock Symbol (e.g. AAPL, GOOG)", 'GOOG')
from datetime import datetime, timedelta

# Get last 10 years of data from today
end = datetime.today()
start = end - timedelta(days=365 * 10)
start_str = start.strftime('%Y-%m-%d')
end_str = end.strftime('%Y-%m-%d')

data = yf.download(stock, start=start_str, end=end_str)

st.subheader('🔍 Raw Stock Data (Most Recent First)')
st.write(data.reset_index().iloc[::-1])

# split the data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(data_train)

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

#############################################

# Predict tomorrow's price using the last 100 days from the original data
st.subheader("Tomorrow's Predicted Price")

# Get the last 100 days of closing prices from the full dataset and scale them
last_100_days = data.Close[-100:]
final_input = scaler.transform(np.array(last_100_days).reshape(-1, 1))

# Reshape to match model input
final_input = final_input.reshape(1, 100, 1)

# Predict
predicted_price_scaled = model.predict(final_input)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

# Show result
st.write(f"Predicted Price for Next Day: **${predicted_price[0][0]:.2f}**")


#############################################################################

st.subheader("Future Forecast")
n_days = st.number_input("Enter number of days to forecast", min_value=1, max_value=30, value=7)
# Get the last 100 days and scale them
last_100 = data.Close[-100:]
input_seq = scaler.transform(np.array(last_100).reshape(-1, 1)).tolist()

# Predict next n_days
future_predictions = []

for _ in range(n_days):
    x_input = np.array(input_seq[-100:]).reshape(1, 100, 1)
    pred_scaled = model.predict(x_input)
    input_seq.append(pred_scaled[0])
    future_predictions.append(pred_scaled[0])

# Inverse transform predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future dates
last_date = data.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days)

# Plot
st.subheader(f"{n_days}-Day Forecast")
fig5 = plt.figure(figsize=(8, 6))
plt.plot(future_dates, future_predictions, marker='o', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'Forecast for Next {n_days} Days')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig5)

st.subheader('Price vs Moving Average of 50 days')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)



x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

predict = scaler.inverse_transform(predict.reshape(-1, 1))
y = scaler.inverse_transform(y.reshape(-1, 1))

#fig1

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label= 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)