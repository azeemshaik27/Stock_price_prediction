# Stock Prediction App - Deployment TODO

## Approved Plan: Vercel Deployment
**Goal:** Production-ready Flask app with pre-trained models, optional training, serverless Vercel deploy.

### Step 1: Create TODO.md [IN PROGRESS]
- [x] Plan confirmed (Vercel target)

### Step 2: Update requirements.txt
- [x] Add gunicorn, torch

### Step 3: Update stock-prediction/app.py
- Production config (host, port, debug env vars)
- Skip train if models exist in /run
- Create data/model/results dirs safely
- Fix duplicate metrics code
- Production config (host, port, debug env vars)
- Skip train if models exist in /run
- Create data/model/results dirs safely
- Fix duplicate metrics code

### Step 4: Add path params to predict modules
- Update predict.py, predict_lstm.py to accept data_dir, model_dir

### Step 5: Create Vercel config files
- [x] vercel.json
- [x] README.md
- [x] .env.example

### Step 6: Pre-train default models
- Run `python stock-prediction/train.py --ticker=AAPL`
- Run `python stock-prediction/train_lstm.py --ticker=AAPL`
- Run `python stock-prediction/train.py --ticker=AAPL`
- Run `python stock-prediction/train_lstm.py --ticker=AAPL`

### Step 7: Test locally
- `pip install -r requirements.txt`
- `flask --app stock-prediction/app run --host=0.0.0.0 --port=3000`
- Test /run without retraining

### Step 8: Deploy to Vercel
- `npm i -g vercel`
- `vercel --prod`

**Next:** Step 2 - Edit requirements.txt
