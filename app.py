import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#model = load_model('C:\Users\kasub\OneDrive\Desktop\SimpleStockMarketPredictor\Stock Prediction Model.keras')
model = load_model('Stock Prediction Model.keras')


st.title('📈 AI Stock Price Predictor')

stock = st.text_input("Enter Stock Symbol (e.g. AAPL, GOOG)", 'GOOG')
stock = stock.upper()

try:
    test_data = yf.download(stock, period="1d")
    if test_data.empty:
        st.warning("⚠️ Invalid stock symbol. Please enter a valid ticker (e.g. AAPL, GOOG, MSFT).")
        st.stop()  # Prevents the rest of the code from running
except Exception as e:
    st.error("❌ Error fetching stock data. Please check your internet connection or try a different symbol.")
    st.stop()

from datetime import datetime, timedelta

# Get last 10 years of data from today
end = datetime.today()
start = end - timedelta(days=365 * 10)
start_str = start.strftime('%Y-%m-%d')
end_str = end.strftime('%Y-%m-%d')

#download data
data = yf.download(stock, start=start_str, end=end_str)


# split the data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(data_train)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Prepare data for model prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

predict = scaler.inverse_transform(predict.reshape(-1, 1))
y = scaler.inverse_transform(y.reshape(-1, 1))

#############################################

st.subheader("🔮 Tomorrow's Predicted Price")
last_100_days = data.Close[-100:]
final_input = scaler.transform(np.array(last_100_days).reshape(-1, 1))
final_input = final_input.reshape(1, 100, 1)
predicted_price_scaled = model.predict(final_input)
predicted_price = scaler.inverse_transform(predicted_price_scaled)
st.write(f"Predicted Price for Next Day: **${predicted_price[0][0]:.2f}**")

############################################################################
st.subheader("📊 Forecast Future Prices")
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

import altair as alt

st.subheader(f"{n_days}-Day Forecast (Interactive)")

# Build Altair chart
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': future_predictions.flatten()
})

# Calculate axis limits
y_min_forecast = float(forecast_df['Predicted Price'].min())
y_max_forecast = float(forecast_df['Predicted Price'].max())
x_min_forecast = pd.to_datetime(forecast_df['Date'].min())
x_max_forecast = pd.to_datetime(forecast_df['Date'].max())

chart_forecast = alt.Chart(forecast_df).mark_line(point=True).encode(
    x=alt.X('Date:T', title='Date', scale=alt.Scale(domain=[x_min_forecast, x_max_forecast])),
    y=alt.Y('Predicted Price:Q', title='Price ($)', scale=alt.Scale(domain=[y_min_forecast, y_max_forecast])),
    tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Predicted Price:Q', format='.2f')]
).properties(
    height=400,
    title=f"{stock.upper()} - {n_days}-Day Forecast"
).interactive()

st.altair_chart(chart_forecast, use_container_width=True)

####################################################################

st.subheader('Last 30 Days + Future Forecast')
# Combine recent prices + forecast
combined_prices = pd.concat([
    pd.DataFrame({
        'Date': list(data.index[-30:]),
        'Price': data.Close[-30:].values.flatten()
    }),
    pd.DataFrame({
        'Date': pd.date_range(data.index[-1] + timedelta(days=1), periods=n_days),
        'Price': future_predictions.flatten()
    })
])

fig_combined = plt.figure(figsize=(10, 5))
plt.plot(combined_prices['Date'], combined_prices['Price'], marker='o')
plt.axvline(x=data.index[-1], color='gray', linestyle='--', label='Forecast Start')
plt.title(f"{stock.upper()} - Last 30 Days + Next {n_days} Days Forecast")
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig_combined)

st.subheader('Last 30 Days + Future Forecast (Interactive)')

combined_chart_df = pd.DataFrame({
    'Date': combined_prices['Date'],
    'Price': combined_prices['Price']
})

# Calculate axis limits
y_min_combined = float(combined_chart_df['Price'].min())
y_max_combined = float(combined_chart_df['Price'].max())
x_min_combined = pd.to_datetime(combined_chart_df['Date'].min())
x_max_combined = pd.to_datetime(combined_chart_df['Date'].max())

chart_combined = alt.Chart(combined_chart_df).mark_line(point=True).encode(
    x=alt.X('Date:T', title='Date', scale=alt.Scale(domain=[x_min_combined, x_max_combined])),
    y=alt.Y('Price:Q', title='Price ($)', scale=alt.Scale(domain=[y_min_combined, y_max_combined])),
    tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Price:Q', format='.2f')]
).properties(
    height=400,
    title=f"{stock.upper()} - Last 30 Days + Next {n_days} Days Forecast"
).interactive()

st.altair_chart(chart_combined, use_container_width=True)

#############################################################################

# Date range dropdown for plotting historical trend
st.subheader('📆 Trend Price Chart')
# Dropdown for time ranges
time_ranges = {
    "1 Day": 1,
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "5 Years": 1825,
    "10 Years": 3650,
    "Max": None
}

selected_range = st.selectbox("Select time range:", list(time_ranges.keys()), index=5)

# Filter data based on selected range
if time_ranges[selected_range] is None:
    filtered_data = data
else:
    #cutoff_date = data.index[-1] - timedelta(days=time_ranges[selected_range])
    cutoff_date = pd.Timestamp.today() - timedelta(days=time_ranges[selected_range])
    filtered_data = data[data.index >= cutoff_date]

# Plot using matplotlib
fig_hist = plt.figure(figsize=(8, 5))
plt.plot(filtered_data.index, filtered_data['Close'], label='Closing Price')
plt.title(f"{stock.upper()} - {selected_range} Trend")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True)
plt.legend()
st.pyplot(fig_hist)

import altair as alt

st.subheader('📆 Trend Price Chart (Interactive)')

if filtered_data.empty or len(filtered_data) < 2:
    st.warning("Not enough historical data for this range.")
else:
    chart_data = filtered_data.reset_index()[['Date', 'Close']]
    
    # Convert to compatible types
    x_min = pd.to_datetime(chart_data['Date'].min())
    x_max = pd.to_datetime(chart_data['Date'].max())
    y_min = float(chart_data['Close'].min())
    y_max = float(chart_data['Close'].max())

    chart = alt.Chart(chart_data).mark_line().encode(
        x=alt.X('Date:T', title='Date', scale=alt.Scale(domain=[x_min, x_max])),
        y=alt.Y('Close:Q', title='Price ($)', scale=alt.Scale(domain=[y_min, y_max])),
        tooltip=[
            alt.Tooltip('Date:T', title='Date'),
            alt.Tooltip('Close:Q', title='Price ($)', format=".2f")
        ]
    ).properties(
        title=f"{stock.upper()} - {selected_range} Trend",
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

########################################################################

# Display stock data
st.subheader('🔍 Raw Stock Data')
st.write(data.reset_index().iloc[::-1])

#####################################################################

# Original vs Predicted Prices
st.subheader('📉 Model Performance on Historical Test Data')
fig4 = plt.figure(figsize=(8, 5))
plt.plot(y, 'g', label='Actual Price')
plt.plot(predict, 'r', label='Model Prediction')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(fig4)

st.markdown("---")
st.markdown(
    """
    #### 📘 Disclaimer  
    This application is intended for **educational and informational purposes only**.  
    It does not constitute financial advice, investment guidance, or a recommendation to buy or sell any stock.

    Stock prices are influenced by a wide range of real-world factors — including economic conditions, company news, government policies, and global events — that cannot be fully captured by machine learning models trained on historical data.

    Please do your own research and consult with a qualified financial advisor before making any investment decisions.  
    
    The predictions shown are based on historical data and machine learning models, which may not accurately reflect future performance.
    """,
    unsafe_allow_html=True
)

#st.subheader('Price vs Moving Average of 50 days')
#ma_50_days = data.Close.rolling(50).mean()
#fig1 = plt.figure(figsize=(8,6))
#plt.plot(ma_50_days, 'r')
#plt.plot(data.Close, 'g')
#plt.show()
#st.pyplot(fig1)
