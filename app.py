import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.index = pd.to_datetime(data.index)  # Convert index to DateTimeIndex
    data.index = data.index.tz_localize("UTC").tz_convert("UTC")  # Add this line to localize and convert the timezone
    return data


# Function to preprocess data and create features
def preprocess_data(data):
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Day_of_week'] = data['Date'].dt.dayofweek
    return data

# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the model
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse}')

    return model, scaler

# Function to predict future prices
def predict_future_prices(model, scaler, X_future):
    X_future_scaled = scaler.transform(X_future)
    future_predictions = model.predict(X_future_scaled)
    return future_predictions

# Function to preprocess data and create features for Open prices
def preprocess_data_open(data):
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Day_of_week'] = data['Date'].dt.dayofweek
    return data

# Function to train the model for Open prices
def train_model_open(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the model
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error for Open prices: {mse}')

    return model, scaler

# Function to predict future prices for Open prices
def predict_future_open_prices(model, scaler, X_future):
    X_future_scaled = scaler.transform(X_future)
    future_open_predictions = model.predict(X_future_scaled)
    return future_open_predictions

# Streamlit UI
st.title("Stock Price Forecasting App")

# Get user input for the stock ticker and date range
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.date_input("Select Start Date to Analyze:", pd.to_datetime('2022-01-01'))
end_date = st.date_input("Select End Date to Analyze:", pd.to_datetime('2022-12-31'))

# Fetch historical data for Close prices
stock_data = get_stock_data(ticker, start_date, end_date)

# Preprocess data for Close prices
processed_data = preprocess_data(stock_data)

# Define features and target variable for Close prices
features = ['Year', 'Month', 'Day', 'Day_of_week']
target = 'Close'

X = processed_data[features]
y = processed_data[target]

# Train the model for Close prices
trained_model, trained_scaler = train_model(X, y)

# Slider for selecting the number of previous days
num_prev_days = st.slider("Select the number of previous days to include", min_value=1, max_value=len(stock_data), value=30)

# Slider for selecting the number of future days to predict
num_future_days = st.slider("Select the number of future days to predict", min_value=1, max_value=365, value=30)

# Ask user for today's date
today = st.date_input("Select Today's Date:", pd.to_datetime('2023-12-14'))  # Change this to the current date

# Prediction for the next 'num_future_days'
previous_data = processed_data.iloc[-num_prev_days:]
future_dates = pd.date_range(today + pd.DateOffset(days=1), periods=num_future_days, freq='B')  # Assuming 20 business days in a month

X_future = pd.DataFrame({'Year': future_dates.year,
                         'Month': future_dates.month,
                         'Day': future_dates.day,
                         'Day_of_week': future_dates.dayofweek})

future_predictions = predict_future_prices(trained_model, trained_scaler, X_future)

# Display predictions for Close prices
future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})

# Combine with previous data for chart
combined_data = pd.concat([previous_data, future_predictions_df])

# Fetch historical data for Open prices
stock_data_open = get_stock_data(ticker, start_date, end_date)

# Preprocess data for Open prices
processed_data_open = preprocess_data_open(stock_data_open)

# Define features and target variable for Open prices
target_open = 'Open'
X_open = processed_data_open[features]
y_open = processed_data_open[target_open]

# Train the model for Open prices
trained_model_open, trained_scaler_open = train_model_open(X_open, y_open)

# Prediction for the next 'num_future_days'
future_open_predictions = predict_future_open_prices(trained_model_open, trained_scaler_open, X_future)

# Display predictions for Open prices
future_open_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Open': future_open_predictions})

# Combine with previous data for chart
combined_data_open = pd.concat([processed_data_open.iloc[-num_prev_days:], future_open_predictions_df])

# Display predictions in Streamlit
st.subheader("Predicted Prices for the Next 'num_future_days' Days:")
st.table(pd.concat([future_predictions_df.set_index('Date')['Predicted_Close'], future_open_predictions_df.set_index('Date')['Predicted_Open']], axis=1))

# Display line chart for Open and Close prices
st.subheader("Open and Close Prices Over Time:")
st.line_chart(pd.concat([combined_data.set_index('Date')[['Close', 'Predicted_Close']], combined_data_open.set_index('Date')[['Open', 'Predicted_Open']]], axis=1))
