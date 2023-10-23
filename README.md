# Random-Forest-Classifier-Stock-Machine-Learning-
Machine Learning algorithm that considers historical closing price data, 5-day MA and 20-day MA to make predictions on when to buy and sell a stock. 

## Description:
This project employs the power of the Random Forest Classifier to predict the directional movements of stock prices. By analyzing historical stock price data, the model seeks to determine whether a stock will move upwards or downwards on a given day.

## Key Features:
- Data Collection: Downloads historical stock price data for specified tickers.
- Data Preprocessing: Calculates stock returns and prepares data for modeling.
- Feature Engineering: Uses technical indicators such as Simple Moving Averages (SMA) as features for the model.
- Model Training & Evaluation: Trains a Random Forest Classifier and evaluates its performance in terms of accuracy.
- Trading Strategy Design: Designs a basic trading strategy based on the model's predictions.
- Backtesting: Implements backtesting to assess the potential performance of the trading strategy on historical data.

## Technical Details:
- Languages & Libraries: Python, pandas, yfinance, scikit-learn, plotly.
- Machine Learning Technique: Random Forest Classifier.
- Data Source: Stock data sourced from Yahoo Finance through the yfinance library.

## Model Training and Test Accuracy
![image](https://github.com/MorrisAdam0/Random-Forest-Classifier-Stock-Machine-Learning-/assets/115980966/1aa7eba0-b71f-4fe9-ac60-b9afac3886f4)

## Web Plotly Interface of Buy and Sell Signals and Cumulative Returns
![image](https://github.com/MorrisAdam0/Random-Forest-Classifier-Stock-Machine-Learning-/assets/115980966/7c384710-ff4b-4d74-938f-a785f4cc25c5)

## Visualisation of Model Profit and Loss
![image](https://github.com/MorrisAdam0/Random-Forest-Classifier-Stock-Machine-Learning-/assets/115980966/30c5574a-89e6-4027-87d4-2e025a5b662a)
