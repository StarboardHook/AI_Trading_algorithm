import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

class Backtester:
    def __init__(self, model, scaler, initial_balance=1000):
        self.model = model
        self.scaler = scaler
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []  # Stores open positions
        self.trade_log = []  # Stores completed trades

    def predict_signal(self, data, sequence_length=50):
        """Generate buy/sell signals using the trained LSTM model."""
        signals = []
        
        for i in range(sequence_length, len(data)):
            # Prepare input data for prediction
            input_data = data[i-sequence_length:i][['close']]
            input_data_scaled = self.scaler.transform(input_data)
            input_data_scaled = np.reshape(input_data_scaled, (1, input_data_scaled.shape[0], input_data_scaled.shape[1]))
            
            # Get model prediction (next day's price prediction)
            predicted_price = self.model.predict(input_data_scaled)
            predicted_price = self.scaler.inverse_transform(predicted_price.reshape(-1, 1))[0][0]
            
            current_price = data.iloc[i]['close']
            print(f"Date: {data.index[i]}, Predicted: {predicted_price}, Actual: {current_price}")
            # Generate buy/sell signal
            if predicted_price > current_price:
                signals.append('buy')
            elif predicted_price < current_price:
                signals.append('sell')
            else:
                signals.append('hold')  # No signal if price remains the same

        return signals

    def simulate_trades(self, data, signals):
        signal_index = 0 # add a counter to keep track of signals index.
        for index, row in data.iterrows():
            if signal_index < len(signals): # make sure we do not go out of bounds.
                signal = signals[signal_index]
                price = row['close']

            if signal == 'buy':
                self.positions.append(price)
            elif signal == 'sell' and self.positions:
                buy_price = self.positions.pop(0)  # FIFO approach
                profit = price - buy_price
                self.trade_log.append(profit)
                self.balance += profit
            signal_index += 1 # increment signal index.
        print(f"Trade: {signal}, Price: {price}, Balance: {self.balance}")

    def calculate_performance_metrics(self):
        returns = np.array(self.trade_log)
        avg_return = np.mean(returns) if len(returns) > 0 else 0
        std_dev = np.std(returns) if len(returns) > 0 else 1
        sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0
        
        peak = self.initial_balance
        max_drawdown = 0
        for trade in self.trade_log:
            self.balance += trade
            if self.balance > peak:
                peak = self.balance
            drawdown = (peak - self.balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        win_rate = (sum([1 for p in self.trade_log if p > 0]) / len(self.trade_log)) * 100 if self.trade_log else 0
        profit_factor = (sum([p for p in self.trade_log if p > 0]) / -sum([p for p in self.trade_log if p < 0])) if any(p < 0 for p in self.trade_log) else float('inf')
        
        return {
        'Final Balance': self.balance,  # The total balance at the end of the backtest.
        'Sharpe Ratio': sharpe_ratio,  # A measure of risk-adjusted return.
        'Max Drawdown': max_drawdown,  # The largest peak-to-valley drop in account balance.
        'Win Rate (%)': win_rate,  # The percentage of profitable trades.
        'Profit Factor': profit_factor  # The ratio of gross profit to gross loss.
        }

def load_model_and_scaler():
    model = tf.keras.models.load_model('./models/lstm_model.h5')
    scaler = joblib.load('./models/scaler.pkl')
    return model, scaler

if __name__ == "__main__":
    # Load the trained model and scaler
    model, scaler = load_model_and_scaler()
    
    # Load historical stock data
    data = pd.read_csv('./data/Out_of_Sample_data.csv', index_col=0, parse_dates=True)
    print("Backtest Data Range:", data.index.min(), "to", data.index.max())
    # Initialize backtester
    backtester = Backtester(model, scaler)
    
    # Generate buy/sell signals
    signals = backtester.predict_signal(data)
    
    # Simulate trades based on generated signals
    backtester.simulate_trades(data, signals)
    
    # Calculate and display performance metrics
    results = backtester.calculate_performance_metrics()
    print("Backtest Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # Save the backtest results to a CSV file
    pd.DataFrame([results]).to_csv('./data/backtest_results/results.csv', index=False)
