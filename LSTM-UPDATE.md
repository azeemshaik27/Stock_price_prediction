# LSTM + Bright Green + Better Graph Plan

✅ **Approved:** Proceed with bright green theme + LSTM model + better graph.

**Information Gathered:**
- Frontend: templates/index.html (toggle ready).
- Backend: app.py (Flask), train.py (ARIMA), predict.py.
- Data: data/stock_data.csv (yfinance).
- Model: model/arima_model.pkl.
- requirements.txt: Add torch/tensor for LSTM.
- Logs: ARIMA working.

**Plan (file-level):**
1. **Frontend (index.html):** Bright green light theme (#22C55E accent, #F0FDF4 bg). Model selector (ARIMA/LSTM toggle). Better graph (interactive Plotly.js CDN).
2. **Backend:**
   - app.py: Add /run_lstm endpoint, model param.
   - train_lstm.py: New file for LSTM train/save (torch.nn.LSTM).
   - predict_lstm.py: New for LSTM predict.
3. **requirements.txt:** Add torch, plotly.
4. Update UI JS: Send model= 'arima'|'lstm', render Plotly chart from JSON data.

**Dependent Files:**
- Edit: templates/index.html, app.py, requirements.txt.
- Create: train_lstm.py, predict_lstm.py.
- model/lstm_model.pt.

**Progress:**
✅ pip torch/plotly installed.
✅ Files: train_lstm.py, predict_lstm.py, app.py (endpoints /run /run_lstm), index.html (model toggle, Plotly, bright green).
✅ LSTM training running (RELIANCE.NS).

**Status:** Complete! Test at http://localhost:5000  
- ARIMA: Works.
- LSTM: Train complete soon, toggle model, interactive graphs.

Ready for use.
