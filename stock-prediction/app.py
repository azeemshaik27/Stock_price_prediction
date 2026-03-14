import io
import os
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no popup window)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template, request, jsonify

import train as trainer
import predict as predictor
import train_lstm
import predict_lstm

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Chart Builder ─────────────────────────────────────────────────────────────
def build_chart(ticker: str, df_forecast: pd.DataFrame) -> str:
    """
    Creates a price chart with:
      - Historical prices (last 6 months shown for clarity)
      - Forecast line
      - 95% confidence interval shading
    Returns chart as base64 string (embedded in HTML, no file needed).
    """
    # Load historical
    hist = pd.read_csv("data/stock_data.csv", index_col=0, parse_dates=True)
    hist = hist["Close"].squeeze().iloc[-120:]   # last ~6 months

    # Parse forecast dates
    df_forecast = df_forecast.copy()
    df_forecast["date"] = pd.to_datetime(df_forecast["date"])

    # ── Style ──
    BG      = "#F9F8F5"
    FG      = "#1C1C1E"
    MUTED   = "#AAAAAA"
    ACCENT  = "#2C5F8A"
    GREEN   = "#2E7D55"
    GRID    = "#EBEBEB"

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Historical line
    ax.plot(hist.index, hist.values,
            color=FG, lw=1.3, label="Historical", zorder=3)

    # Forecast line
    ax.plot(df_forecast["date"], df_forecast["forecast"],
            color=ACCENT, lw=1.5, ls="--", label="Forecast", zorder=4)

    # Confidence interval
    ax.fill_between(
        df_forecast["date"],
        df_forecast["lower_95"],
        df_forecast["upper_95"],
        color=GREEN, alpha=0.12, label="95% Confidence Interval"
    )

    # Divider
    ax.axvline(df_forecast["date"].iloc[0], color=MUTED, lw=0.8, ls=":")

    # Last actual price annotation
    last_price = hist.values[-1]
    ax.annotate(f"  ₹{last_price:.2f}",
                xy=(hist.index[-1], last_price),
                fontsize=8, color=MUTED)

    # End forecast annotation
    end_price = df_forecast["forecast"].iloc[-1]
    ax.annotate(f"  ₹{end_price:.2f}",
                xy=(df_forecast["date"].iloc[-1], end_price),
                fontsize=8, color=ACCENT)

    # Formatting
    ax.set_title(f"{ticker} · Price Forecast",
                 fontsize=10, fontweight="bold", color=FG, pad=10)
    ax.set_ylabel("Price (INR ₹)", fontsize=8, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_b64

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")

@app.route("/arima")
def arima():
    return render_template("index_arima.html")

@app.route("/lstm")
def lstm():
    return render_template("index_lstm.html")


@app.route("/run", methods=["POST"])
def run_pipeline():
    data       = request.get_json()
    ticker     = data.get("ticker", "AAPL").upper().strip()
    start      = data.get("start",  "2021-01-01")
    end        = data.get("end",    "2024-12-31")
    days       = int(data.get("days", 30))
    model_type = data.get("model", "arima")

    forecasts = {}
    metrics_list = []
    
    model_type = data.get("model", "arima")
    
    forecasts = {}
    metrics_list = []
    
    if model_type == "arima":
        trainer.main(ticker, start, end)
        df_forecast, metrics = predictor.run(days=days)
        forecasts["arima"] = df_forecast.to_dict('records')
        metrics_list.append(metrics)
        table = df_forecast.to_dict('records')
    elif model_type == "lstm":
        train_lstm.main(ticker, start, end)
        df_forecast, metrics = predict_lstm.run(days=days)
        forecasts["lstm"] = df_forecast.to_dict('records')
        metrics_list.append(metrics)
        table = df_forecast.to_dict('records')
    
    chart_b64 = ""
    if forecasts:
        first_key = list(forecasts.keys())[0]
        df_f = pd.DataFrame(forecasts[first_key])
        chart_b64 = build_chart(ticker, df_f)
    
    return jsonify({
        "status": "ok",
        "chart": chart_b64,
        "table": table,
        "metrics": metrics_list[0] if metrics_list else {},
        "ticker": ticker,
        "days": days
    })

# ── Start Server ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Stock Price Prediction")
    print("  Open browser at: http://localhost:5000\n")
    app.run(debug=True, port=5000)
