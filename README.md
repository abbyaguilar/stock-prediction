#Stock Price Forecasting App

Overview
This is a Python application for forecasting stock prices using historical data and machine learning techniques. The app is built with Streamlit and utilizes the yfinance library for fetching stock data and scikit-learn for building a Gradient Boosting Regressor model.

Installation

Clone the Repository

  git clone https://github.com/yourusername/stock-predictor.git
  cd stock-predictor
  
Create Virtual Environment

  python -m venv venv
  
Activate Virtual Environment

On Windows:

  venv\Scripts\activate
  
On macOS/Linux:
  
  source venv/bin/activate
  
Install Dependencies

  pip install -r requirements.txt
  
Usage

  streamlit run stock_predictor_app.py
  
The app will be accessible in your web browser at http://localhost:8501.

Input Parameters

Enter the stock ticker (e.g., AAPL).
Select the start and end dates for analysis.
Adjust sliders for the number of previous days to consider and the number of future days to predict.
View Predictions

The app will display predicted close and open prices for the selected stock.
A line chart showing historical and predicted open and close prices over time will be displayed.

Data and Model

The app fetches historical stock data using the yfinance library.
Features such as year, month, day, and day of the week are extracted from the date to train the model.
The model is a Gradient Boosting Regressor, and its hyperparameters are optimized using GridSearchCV.
Predictions are made for both close and open prices.


Notes

Ensure you have an active internet connection to fetch stock data.
The app assumes 20 business days in a month for predicting future dates.

Libraries Used
yfinance
pandas
scikit-learn
streamlit


Disclaimer
This app provides stock price predictions for informational purposes only. Predictions may not be accurate, and users should not solely rely on them for financial decisions. Always consult with a financial advisor before making investment decisions.
