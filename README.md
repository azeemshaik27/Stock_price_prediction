# Stock Price Prediction App

## Local Run
```bash
pip install -r requirements.txt
python stock-prediction/train.py --ticker=AAPL
python stock-prediction/train_lstm.py --ticker=AAPL
python stock-prediction/app.py
```
Visit http://localhost:5000

## Vercel Deployment (Python 3.10.12)
Uses `runtime.txt`. 

1. `git init && git add . && git commit -m "Deploy ready" && git push` GitHub.
2. vercel.com → New Project → Import.
3. Set env: FLASK_DEBUG=false, TRAIN_ON_DEMAND=false.

Live: Serverless, fast predictions w/ pre-trained AAPL models.

Live demo: Models skip training if exist – fast predictions.
