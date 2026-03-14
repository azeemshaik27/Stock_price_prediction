import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib  # for scaler if needed, but we save scaler in train

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_model():
    if not os.path.exists("model/lstm_model.pt"):
        raise FileNotFoundError("Run python train_lstm.py first")
    
    checkpoint = torch.load("model/lstm_model.pt", map_location='cpu', weights_only=False)
    model = LSTMModel()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    scaler = checkpoint['scaler']
    print("[lstm_predict] ✓ LSTM loaded")
    return model, scaler, checkpoint['ticker']

def load_data():
    df = pd.read_csv("data/stock_data.csv", index_col=0, parse_dates=True)
    series = df['Close'].values.reshape(-1,1)
    print(f"[lstm_predict] ✓ Data loaded {len(series)} rows")
    return series

def predict(model, scaler, series, days=30, seq_len=60):
    model.eval()
    last_seq = scaler.transform(series[-seq_len:].reshape(-1,1))
    last_seq = torch.FloatTensor(last_seq).unsqueeze(0)
    
    forecasts = []
    current_seq = last_seq.clone()
    
    with torch.no_grad():
        for _ in range(days):
            pred = model(current_seq)
            forecasts.append(pred.item())
            # Shift sequence
            new_seq = torch.cat((current_seq[:,1:,:], pred.unsqueeze(0)), dim=1)
            current_seq = new_seq
    
    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1,1)).flatten()
    
    # Business days
    last_date = pd.to_datetime(series.index[-1]) if hasattr(series, 'index') else pd.Timestamp.now()
    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=days)
    
    df_forecast = pd.DataFrame({
        'date': future_dates.strftime('%Y-%m-%d'),
        'forecast': forecasts
    })
    
    # Dummy CI for now (LSTM no native CI; use std dev or bootstrap later)
    df_forecast['lower_95'] = df_forecast['forecast'] * 0.95
    df_forecast['upper_95'] = df_forecast['forecast'] * 1.05
    
    df_forecast.to_csv('results/lstm_predictions.csv', index=False)
    print(f"[lstm_predict] ✓ {days}-day forecast ready")
    return df_forecast

def evaluate(model, scaler, series, seq_len=60):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    test_size = min(30, len(series)//10)
    if test_size < 2:
        return {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}
    
    # series is numpy array
    test_data_scaled = scaler.transform(series[-test_size-seq_len:].reshape(-1,1))
    test_data = test_data_scaled[-test_size:]
    
    preds = []
    for i in range(test_size):
        X_i = torch.FloatTensor(test_data_scaled[i:i+seq_len].reshape(1, seq_len, 1))
        pred = model(X_i).item()
        preds.append(pred)
    
    y_test = scaler.transform(series[-test_size:].reshape(-1,1)).flatten()
    
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    
    metrics = {"RMSE": round(rmse,3), "MAE": round(mae,3), "MAPE": round(mape,2)}
    print(f"[lstm_predict] Eval: RMSE={metrics['RMSE']}")
    return metrics

def run(days=30):
    model, scaler, ticker = load_model()
    series = load_data()
    metrics = evaluate(model, scaler, series)
    df_forecast = predict(model, scaler, series, days)
    return df_forecast, metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    df, metrics = run(args.days)
    print(df.head())

