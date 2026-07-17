# AI-Driven Regime-Aware Pairs Trading in Healthcare

**Statistical Arbitrage · Graph Attention Network · Hidden Markov Model · PPO Reinforcement Learning · Python**

> FE-670 Final Project — MS Financial Engineering, Stevens Institute of Technology (Dec 2025)
> Authors: Swara Dave · Parth Vora · Robin Singh

---

## 📌 Overview

Classic pairs trading assumes stable relationships between assets — but markets switch between calm and crisis regimes, and naïve Z-score rules often over-trade in noisy environments. This project builds an end-to-end AI-driven, regime-aware pairs trading framework in the U.S. healthcare and biopharma sector combining:

1. **Statistical screening** — Johansen cointegration tests across 741 pairs in a 39-asset universe
2. **Graph Attention Network (GAT)** — learns a data-driven similarity score over all assets using network structure
3. **Hidden Markov Model (HMM)** — detects 3 market regimes (Calm, Normal, Volatile) to condition trade execution
4. **PPO Reinforcement Learning agent** — executes mean-reversion trades in a custom gym environment with transaction costs

---

## 📊 Key Results

- Screened **741 pairs** across 39 healthcare equities → **36 cointegrated candidates** (Johansen p < 0.05)
- GAT Spearman rank correlation between model scores and realized P&L: **ρ = 0.5629 (p = 0.0004)**
- PPO agent achieved **94–96% win rates** across COVID and post-COVID regimes with controlled drawdowns
- Top performing pairs: **BNTX–RGEN** and **AMGN–LLY** — high win rates and stable mean-reverting spreads
- GNN Precision-at-K: top 5 and top 10 ranked pairs had substantially higher profitable strategy rates than the full set

---

## 🗂️ Data

Daily adjusted close prices from **Feb 6, 2020 to Dec 5, 2025** for 39 U.S. healthcare and biopharma tickers.

Features engineered per asset:
- Log returns, annualized volatility, rolling Sharpe-like ratio
- Pairwise Johansen cointegration p-values
- Cross-sectional average return, volatility, and correlation (HMM inputs)

---

## ⚙️ Methodology

### Pipeline Overview
```
Data & EDA → Cointegration Screening → HMM Regime Detection → GAT Pair Scoring → PPO Execution → Evaluation
```

### 1. Statistical Screening
- Computed correlation matrix and pairwise Johansen cointegration p-values for all 741 pairs
- Filtered to 36 "strong pairs" at p < 0.05 threshold

### 2. Hidden Markov Model (HMM)
- 3-state HMM on universe-level factors: cross-sectional average return, volatility, average pairwise correlation
- Regime labels: **Calm** (low vol, moderate returns), **Normal** (average vol), **Volatile** (elevated variance, extreme moves)
- Regime label fed into PPO state vector — same Z-score has different meaning across regimes

### 3. Graph Attention Network (GAT)
- **Nodes:** 39 assets with features (return, volatility, Sharpe ratio, sector indicators)
- **Edges:** cointegrated pairs + highly correlated pairs (weighted by statistical strength)
- **Architecture:** 2 attention layers → 32-dim hidden space (4 heads) → 128-dim embedding → 16-dim output
- **Training:** binary cross-entropy loss, Adam optimizer, dropout 0.2
- **Output:** dot-product similarity score per pair → ranked list of trading candidates

### 4. PPO Reinforcement Learning Execution
- Custom OpenAI Gym environment per pair
- **State vector:** spread Z-score, spread velocity, HMM regime, current position, unrealized PnL (normalized)
- **Action space:** hold / enter long spread / enter short spread / exit
- **Reward:** PnL change net of transaction costs — penalizes over-trading
- Trained on rolling windows with clipped policy gradients for stable learning

---

## 📁 Repository Structure

```
PairsTrading/
├── 1_Data_Ingestion_&_EDA.ipynb                    # Data loading, EDA, correlation & cointegration analysis
├── 2_Hidden_Markov_Model_(Regime_Detection).ipynb  # HMM regime detection
├── 3_DRL_(PPO).ipynb                               # PPO agent training & backtesting
├── 4_GNN.ipynb                                     # Graph Attention Network pair scoring
└── Results/
    ├── cointegration_pvalues.csv                   # Johansen test p-values for all 741 pairs
    ├── correlation_matrix.csv                      # Pairwise correlation matrix for 39 assets
    ├── strong_pairs.csv                            # 36 cointegrated pairs (p < 0.05)
    ├── hmm_regimes.csv                             # Daily HMM regime labels
    ├── gnn_all_pairs_scores.csv                    # GAT similarity scores for all pairs
    ├── all_pairs_results.csv                       # PPO trading results across all 36 pairs
    └── gnn_precision_metrics.csv                   # GNN validation and precision-at-K metrics
```

## 🚀 How to Run

1. Clone the repo
2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torch-geometric stable-baselines3 gymnasium hmmlearn yfinance
```
3. Run notebooks in order:
   - `1_Data_Ingestion_&_EDA.ipynb` — fetches stock data via `yfinance`, runs EDA and cointegration screening
   - `2_Hidden_Markov_Model_(Regime_Detection).ipynb` — fit HMM and label regimes
   - `4_GNN.ipynb` — train GAT and score pairs
   - `3_DRL_(PPO).ipynb` — train PPO agents and evaluate

> **Note:** Stock data is pulled automatically via `yfinance` inside the first notebook. No manual data download required. Pre-computed results are available in the `Results/` folder.

## 👤 Authors

**Swara Dave** — MS Financial Engineering, Stevens Institute of Technology
[![LinkedIn](https://img.shields.io/badge/LinkedIn-swara--dave-blue?style=flat&logo=linkedin)](https://linkedin.com/in/swara-dave) [![GitHub](https://img.shields.io/badge/GitHub-swaraaaa-black?style=flat&logo=github)](https://github.com/swaraaaa)
