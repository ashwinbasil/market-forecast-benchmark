# AAPL Forecast: LSTM vs Baselines, Done Properly

Started as MSc Data Analytics dissertation (De Montfort University, 2023): "From Data to Dollars". Single-ticker LSTM on AAPL close price.
This repo rebuilds it with proper validation, baselines, and an ML layer.

## Status
- [x] Phase 0: import original work, untouched (notebooks/00_dissertation_original.ipynb)
- [ ] Phase 1: baselines + fixes (naive, MA, ARIMA, walk-forward, no leakage, seeds)
- [ ] Phase 2: ML layer (XGBoost/LightGBM, SHAP, up/down classifier, backtest with costs)

## Known flaws in original (to fix)
- No baseline model
- Predicts price, not return
- Single 95/5 split, no walk-forward
- 1 epoch, no seed, result not reproducible (doc RMSE 2.54, notebook rerun 2.81)
- Scaler fit on full data before split (leakage)
- Close price only, one ticker
- RMSE only

## Layout
notebooks/ (numbered exploration), src/aapl_forecast/ (reusable code), data/ (local cache, gitignored), reports/ (figures, results), docs/ (original dissertation), tests/

## Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
