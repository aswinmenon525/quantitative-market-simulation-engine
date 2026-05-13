# Quantitative Market Simulation Engine

A Python-based stock market analysis tool that applies **LTI Moving Average filtering** and **Random Process theory** to real Indian stock market data. Built as a mini project for the Random Processes course (BECE207L) at VIT Vellore.

> Demonstrates that signal processing concepts from ECE are directly applicable to financial time-series analysis.

---

## What It Does

- Loads real stock price CSV data (Yahoo Finance)
- Applies a **Moving Average LTI filter** to extract the underlying price trend from noisy daily data
- Analyzes daily returns as a **Gaussian Random Process**
- Computes **Autocorrelation** to verify returns are random (no exploitable pattern)
- Measures **Volatility** (standard deviation) as a risk metric
- Generates BUY / SELL / HOLD signals based on price vs MA divergence
- Produces all 4 analysis graphs combined into a **single output image**

---

## Random Process Concepts Applied

| Concept | Application in Project |
|---------|----------------------|
| LTI Moving Average Filter | Removes daily noise, extracts price trend |
| Gaussian Random Variable | Daily stock returns follow bell curve |
| Autocorrelation R_xx(τ) | Checks if past prices predict future prices |
| Volatility (Std Deviation) | Measures randomness = risk level |
| Low-Pass Filter | Proves MA filter passes slow trends, blocks fast noise |

---

## Output — All 4 Graphs Combined

All graphs are generated as a single combined figure `all_graphs_combined.png`:

| Graph | What It Shows |
|-------|--------------|
| Graph 1 | Stock Price with 20-Day MA Filter — BUY/SELL/HOLD signal |
| Graph 2 | Daily Returns Histogram vs Gaussian Bell Curve |
| Graph 3 | Autocorrelation of Daily Returns |
| Graph 4 | Volatility and Risk Comparison |

---

## Theory Plots

### Plot 1 — Raw Signal vs MA Filtered Signal
Shows how increasing window size W produces smoother output but introduces more lag.

![Raw vs Filtered](plots/plot1_raw_vs_filtered.png)

### Plot 2 — Residual Noise Analysis
Shows what the W=10 filter removed. Bottom panel confirms residual is random with mean ≈ 0.

![Residual Noise](plots/plot2_residual_noise.png)

### Plot 3 — Frequency Response of Moving Average Filter
Proves the MA filter is a Low-Pass LTI System — slow trends pass through, fast noise is blocked.

![Frequency Response](plots/plot3_frequency_response.png)

---

## Live Analysis Results (Zomato vs Swiggy — April 2026)

### Graph 1 — Stock Price with 20-Day MA
![Price Trend](graphs/graph1_price_trend.jpeg)

### Graph 2 — Daily Returns vs Gaussian Bell Curve
![Gaussian Returns](graphs/graph2_returns_gaussian.jpeg)

### Graph 3 — Autocorrelation of Daily Returns
![Autocorrelation](graphs/graph3_autocorrelation.jpeg)

### Graph 4 — Volatility and Risk Comparison
![Volatility](graphs/graph4_volatility_signals.jpeg)

---

## Terminal Output

![Terminal Output 1](terminal/terminal_output1.jpeg)
![Terminal Output 2](terminal/terminal_output2.jpeg)

---

## Sample Results

| Stock | Last Price | MA(20d) | Signal | Risk | Diff% |
|-------|-----------|---------|--------|------|-------|
| Zomato | Rs 243.19 | Rs 231.10 | BUY | HIGH | +5.23% |
| Swiggy | Rs 277.85 | Rs 279.95 | HOLD | MEDIUM | -0.75% |

---

## How to Run

1. Install dependencies:
```bash
pip install pandas numpy matplotlib scipy
```

2. Place your Yahoo Finance CSV files in the same folder as the script

3. Edit `STOCK_FILES` in the script with your filenames:
```python
STOCK_FILES = {
    'zomato' : 'zomato.csv',
    'swiggy' : 'swiggy.csv',
}
```

4. Run:
```bash
python market_simulation.py
```

5. Output saved as `all_graphs_combined.png` in the same folder

---

## Tech Stack

- Python 3 | Thonny IDE
- NumPy | Pandas | Matplotlib | SciPy
- Yahoo Finance CSV data

---

## Project Structure

```
quantitative-market-simulation-engine/
├── market_simulation.py
├── README.md
├── plots/
│   ├── plot1_raw_vs_filtered.png
│   ├── plot2_residual_noise.png
│   └── plot3_frequency_response.png
├── graphs/
│   ├── graph1_price_trend.jpeg
│   ├── graph2_returns_gaussian.jpeg
│   ├── graph3_autocorrelation.jpeg
│   └── graph4_volatility_signals.jpeg
└── terminal/
    ├── terminal_output1.jpeg
    └── terminal_output2.jpeg
```

---

## Disclaimer

> Student academic project — BECE207L Random Processes, VIT Vellore.
> Do not make real investment decisions based on this output.
> Always consult a financial advisor before investing.

---

## Author

Aswin — ECE, VIT Vellore (Batch 2024) | 24BEC0525
Course: Random Processes — BECE207L
