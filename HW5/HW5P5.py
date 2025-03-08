import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

#  stock to analyze
ticker = "TSLA"

# Get data for past 30 days
today = datetime.now().date()
start_date = today - timedelta(days=35)  # Extra days for safety

# Fetch the data
data = yf.download(ticker, start=start_date, end=today)


# Use last 30 days of data (excluding today if available)
if today in data.index.date:
    today_actual = data.loc[data.index.date == today, 'Close'].values[0]
    data = data[data.index.date != today]  # Remove today for prediction
else:
    today_actual = None


data = data.tail(30)

# Create features matrix X with 1, x, x^2, x^3 columns
days = np.array(range(len(data)))
X = np.column_stack((
    np.ones(len(days)),      # w0 (constant term)
    days,                    # w1 (linear term)
    days**2,                 # w2 (quadratic term)
    days**3                  # w3 (cubic term)
))

# Target variable (closing prices)
y = data['Close'].values

# Calculate the weight vector w using the normal equation: w = (X^T X)^(-1) X^T y
X_transpose = X.T
w = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

# Predict today's price (day after the last training day)
next_day = len(data)
X_today = np.array([1, next_day, next_day**2, next_day**3])
today_predicted = X_today.dot(w)

print(today_predicted)
