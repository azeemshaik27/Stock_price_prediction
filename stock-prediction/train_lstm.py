import os
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Folders
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

class StockDataset(Dataset):
    def __init__(self, data, seq_len=60):
        self.data = data.flatten()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        x = x.reshape(self.seq_len, 1)  # [seq, feat=1]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq, feat)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    print(f"[lstm_train] Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()
    df.to_csv('data/stock_data.csv')
    print(f"[lstm_train] ✓ {len(df)} rows")
    return df

def prepare_data(series, seq_len=60, train_split=0.8):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1)).flatten()
    
    split = int(len(scaled) * train_split)
    
    return torch.FloatTensor(scaled[:split]), torch.FloatTensor(scaled[split:]), scaler

def train_model(model, train_loader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"[lstm_train] Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

def main(ticker: str, start: str, end: str):
    print(f"\n{'─'*50}")
    print(f"  LSTM Training [{ticker}]")
    print(f"{'─'*50}")

    df = fetch_data(ticker, start, end)
    series = df['Close']
    
    seq_len = 60
    train_data, test_data, scaler = prepare_data(series, seq_len)
    
    train_dataset = StockDataset(train_data, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = LSTMModel()
    train_model(model, train_loader)
    
    torch.save({
        'model': model.state_dict(),
        'scaler': scaler,
        'ticker': ticker
    }, 'model/lstm_model.pt')
    
    print("[lstm_train] ✓ LSTM saved to model/lstm_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    main(args.ticker, args.start, args.end)
