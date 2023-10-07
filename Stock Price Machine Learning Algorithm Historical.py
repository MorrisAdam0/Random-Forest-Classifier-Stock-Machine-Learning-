#This script trains a machine learning Random Forest model to predict stock prices.

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import plotly.subplots as sp
import plotly.graph_objs as go

# Step 1: Data Collection
ticker = "GOOG"
start_date = "2021-01-01"
end_date = "2023-01-06"
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Step 2: Data Preprocessing
data["Return"] = data["Close"].pct_change()
data.dropna(inplace=True)

# Step 3: Feature Engineering
data["SMA_5"] = data["Close"].rolling(window=5).mean()
data["SMA_20"] = data["Close"].rolling(window=20).mean()

# Step 4: Model Selection and Training
X = data[["SMA_5", "SMA_20"]]
y = (data["Return"] > 0).astype(int)

random_state_var = 42
n_estimators_var = 100
test_size_var = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_var, random_state=random_state_var)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForestClassifier(n_estimators=n_estimators_var, random_state=random_state_var))
])

pipeline.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred_train = pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Step 6: Strategy Design
data["Predicted_Return"] = pipeline.predict(X)
data["Signal"] = data["Predicted_Return"].diff()
data.loc[data["Signal"] > 0, "Position"] = 1
data.loc[data["Signal"] < 0, "Position"] = -1
data["Position"].fillna(0, inplace=True)

# Step 7: Backtesting

data["Strategy_Return"] = data["Position"] * data["Return"]
cumulative_returns = (data["Strategy_Return"] + 1).cumprod()

# Create subplots with 2 rows and 1 column
fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

# Create the first subplot for stock closing price
trace1 = go.Scatter(x=data.index, y=data["Close"], mode='lines', name="Stock Closing Price", line=dict(color="blue"))
fig.add_trace(trace1, row=1, col=1)
fig.update_yaxes(title_text="Stock Closing Price", row=1, col=1)

# Create buy signal markers
buy_positions = data[data["Position"] == 1]
trace2 = go.Scatter(x=buy_positions.index, y=buy_positions["Close"], mode='markers', name="Buy Signal",
                         marker=dict(symbol="triangle-up", size=8, color="green", opacity=0.7))
fig.add_trace(trace2, row=1, col=1)

# Create sell signal markers
sell_positions = data[data["Position"] == -1]
trace3 = go.Scatter(x=sell_positions.index, y=sell_positions["Close"], mode='markers', name="Sell Signal",
                         marker=dict(symbol="triangle-down", size=8, color="red", opacity=0.7))
fig.add_trace(trace3, row=1, col=1)

# Create the second subplot for cumulative returns
trace4 = go.Scatter(x=data.index, y=cumulative_returns, mode='lines', name="Cumulative Returns", line=dict(color="red"))
fig.add_trace(trace4, row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Cumulative Returns", row=2, col=1)


# Customize the layout
fig.update_layout(height=800, width=1000, title_text="Stock Data and Cumulative Returns")

# Show the plot
fig.show()

# Calculate Profits and Losses
data["Trade_Return"] = data["Position"].shift(1) * data["Return"]
data["Cumulative_Trade_Return"] = data["Trade_Return"].cumsum()

# Visualize Profits and Losses
plt.figure(figsize=(12, 6))
plt.bar(data.index, data["Trade_Return"], color=data["Trade_Return"].apply(lambda x: 'g' if x > 0 else 'r'))
plt.xlabel("Date")
plt.ylabel("Profit/Loss")
plt.title("Profits and Losses from Trades")
plt.grid()


# Calculate Total Profit and Total Loss
total_profit = data[data["Trade_Return"] > 0]["Trade_Return"].sum()
total_loss = data[data["Trade_Return"] < 0]["Trade_Return"].sum()

print("Total Profit:", total_profit)
print("Total Loss:", total_loss)

#print the data dataframe in full
pd.set_option('display.max_columns', None)
print(data)

plt.show()

#Saving the parameters
algorithm_parameters = {
    "random_state": random_state_var,
    "n_estimators": n_estimators_var,
    "test_size": test_size_var
}

#creating a dataframe to hold the data
data_to_save = pd.DataFrame(columns=["Ticker", "Start Date", "End Date", "Algorithm Parameters", "Cumulative Returns", "Test Accuracy", "Train Accuracy", 'Profit', 'Loss'])
data_to_save.loc[0] = [ticker, start_date, end_date, algorithm_parameters, cumulative_returns[-1], test_accuracy, train_accuracy, total_profit, total_loss]

#define the custom CSV file name
csv_file_path = "backtest_results.csv"


try:
    # Check if the file exists
    if os.path.isfile(csv_file_path):
        # If the file exists, append the DataFrame
        data_to_save.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        data_to_save.to_csv(csv_file_path, index=False)
        print(f"Data saved to {csv_file_path}")
except PermissionError:
    print(f"PermissionError: Could not save data to {csv_file_path}. Check permissions or choose a different directory.")


