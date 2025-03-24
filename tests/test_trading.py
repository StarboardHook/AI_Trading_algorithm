import pandas as pd
from sklearn.metrics import mean_squared_error
from lstm_model import predict_next_price
import logging

# Logging Setup
logging.basicConfig(
    filename='logs/backtest_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def backtest_model(data, model, scaler, initial_balance=10000):
    """Backtest the model on historical data."""
    balance = initial_balance
    position = 0
    predictions = []
    actual_prices = []
    for i in range(len(data) - 1):
        # Get the model's predicted next price
        predicted_price = model.predict(np.array([data[i]]))
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        predictions.append(predicted_price)

        # Get the actual price
        actual_price = data[i + 1]['close']
        actual_prices.append(actual_price)

        # Buy or sell based on prediction
        if predicted_price > actual_price and balance >= actual_price:
            # Buy
            position = balance / actual_price
            balance = 0
        elif predicted_price < actual_price and position > 0:
            # Sell
            balance = position * actual_price
            position = 0

    # Calculate performance
    final_balance = balance + position * data[-1]['close']
    profit = final_balance - initial_balance
    logging.info(f"Backtest finished. Final balance: {final_balance:.2f}, Profit: {profit:.2f}")
    return final_balance, profit, predictions, actual_prices

# Example usage:
df = pd.read_csv("data/historical_stock_data.csv", parse_dates=True, index_col=0)
X, _, scaler = load_data("data/historical_stock_data.csv")

# Train the model
model = train_model()

# Backtest
final_balance, profit, predictions, actual_prices = backtest_model(X, model, scaler)
print(f"Final balance: {final_balance}, Profit: {profit}")