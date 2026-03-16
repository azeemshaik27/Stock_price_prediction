# Stock Price Prediction App

## Local Run
```bash
pip install -r requirements.txt
python stock-prediction/train.py --ticker=AAPL
python stock-prediction/train_lstm.py --ticker=AAPL
python stock-prediction/app.py
```
Visit http://localhost:5000

## Vercel Deployment
1. Push to GitHub
2. Connect repo to Vercel
3. Deploy (auto-detects Python/Flask)

**Env vars:**
- `FLASK_DEBUG`: false
- `TRAIN_ON_DEMAND`: false (skip train by default)

**Pre-trained models:** Commit `data/stock_data.csv`, `model/*` for instant start.

Live demo: Models skip training if exist – fast predictions.
