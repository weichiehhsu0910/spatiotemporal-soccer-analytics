# Spatiotemporal Soccer Analytics: Continuous xT & (Atomic-)VAEP Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://spatiotemporal-soccer-analytics-d9ozvkrzmqvnoncw2xy2vw.streamlit.app/])
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Live Interactive Dashboard:** [https://spatiotemporal-soccer-analytics-d9ozvkrzmqvnoncw2xy2vw.streamlit.app/](https://spatiotemporal-soccer-analytics-d9ozvkrzmqvnoncw2xy2vw.streamlit.app/)  
> Access the deployed web application to inspect continuous xT pitch surfaces, simulate real-time pass threat ($\Delta\text{xT}$), and benchmark (Atomic-)VAEP valuations across 166 international matches.

An end-to-end spatiotemporal soccer analytics framework and interactive tactical web application built on **615,225 raw match events** (**361,378 SPADL actions** and **542,800 Atomic-SPADL actions**) across **166 matches** from the **FIFA World Cup 2022**, **UEFA Euro 2024**, and **UEFA Euro 2020** (Hudl StatsBomb Open Data).

---

## Key Features & Methodological Highlights

1. **Automated Spatiotemporal ETL Pipeline (`src/01_data_etl.py`):**
   - Ingests remote StatsBomb event streams across 166 matches (48 national teams, 1,254 players).
   - Standardizes coordinates ($105\text{m} \times 68\text{m}$) and converts raw events into both **SPADL** and **Atomic-SPADL** sequential representations.
2. **Continuous Expected Threat (xT) via MDP & 2D Gaussian Smoothing (`src/02_xt_model.py`):**
   - Solves a $16 \times 12$ Markov Decision Process (MDP) via value iteration (converging in 54 iterations at $\epsilon = 10^{-5}$).
   - Applies **2D Gaussian kernel smoothing ($\sigma = 0.85$)** and **2D bilinear interpolation (`RegularGridInterpolator`)** to eliminate discrete cell-boundary step artifacts across 285,362 ball-progressing actions ($r = 0.8795$ with discrete xT).
3. **Supervised (Atomic-)VAEP Machine Learning Pipeline (`src/03_vaep_models.py`):**
   - Constructs 3-action sliding game-state features for **Standard-VAEP** ($361,378 \times 568$) and **Atomic-VAEP** ($542,800 \times 229$).
   - Benchmarks **XGBoost** and **CatBoost** classifiers on an 80/20 match-level stratified split (132 train / 34 test matches), achieving out-of-sample **ROC-AUC of 0.8895 ($P_{\text{score}}$)** and **0.9688 ($P_{\text{concede}}$)**.
4. **Interactive Streamlit Visual Analytics Platform (`app.py`):**
   - **Tab 1:** Multi-Model Player Benchmarking Explorer (Continuous xT, Standard-VAEP, Offensive/Defensive VAEP, and Atomic-VAEP).
   - **Tab 2:** Match Trajectory Renderer (`mplsoccer`) & 120-Minute Cumulative Action-Value Momentum Timelines.
   - **Tab 3:** Real-Time Spatial Pass-Threat ($\Delta\text{xT}$) Calculator comparing discrete grid jumps against continuous smoothed surfaces.

---

## Out-of-Sample Model Benchmarks (34 Unseen Test Matches)

| Framework | Learner | Target | Test ROC-AUC | Test Brier Score | Test Log-Loss |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Standard-VAEP** | XGBoost | `scores` | 0.8122 | 0.00673 | 0.03661 |
| **Standard-VAEP** | CatBoost | `scores` | **0.8209** | **0.00668** | **0.03613** |
| **Standard-VAEP** | XGBoost | `concedes` | 0.8753 | 0.00179 | 0.01054 |
| **Standard-VAEP** | CatBoost | `concedes` | **0.8769** | **0.00179** | **0.01051** |
| **Atomic-VAEP** | XGBoost | `scores` | **0.8895** | 0.00473 | 0.02502 |
| **Atomic-VAEP** | CatBoost | `scores` | 0.8871 | **0.00467** | **0.02480** |
| **Atomic-VAEP** | XGBoost | `concedes` | 0.9591 | 0.00093 | 0.00531 |
| **Atomic-VAEP** | CatBoost | `concedes` | **0.9688** | **0.00093** | **0.00526** |

---

## Repository Structure

```text
spatiotemporal-soccer-analytics/
├── app.py                                  # Interactive Streamlit visual analytics web application
├── requirements.txt                        # Python package dependencies
├── README.md                               # Project documentation and empirical benchmarks
├── src/
│   ├── 01_data_etl.py                      # StatsBomb Open Data ingestion & (Atomic-)SPADL conversion
│   ├── 02_xt_model.py                      # MDP Expected Threat (xT), 2D Gaussian smoothing & interpolation
│   ├── 03_vaep_models.py                   # Standard-VAEP & Atomic-VAEP (XGBoost & CatBoost) pipeline
│   └── 04_generate_match_case_study.py     # 2022 World Cup Final momentum & trajectory figure generator
├── models/
│   ├── xt_discrete_matrix.npy              # Converged 16x12 MDP xT surface array
│   ├── xt_smoothed_matrix.npy              # 2D Gaussian-smoothed xT surface array
│   ├── vaep_model_benchmarks.csv           # Out-of-sample ROC-AUC, Brier Score, and Log-Loss metrics
│   ├── fig1_xt_surface_comparison.png      # Discrete vs. Continuous xT valuation surface plot
│   ├── fig2_vaep_xt_benchmark.png          # Multi-model ROC-AUC & player valuation scatter plots
│   └── fig3_world_cup_final_case_study.png # 2022 World Cup Final cumulative VAEP & trajectory plot
└── data/
    ├── games.parquet                       # Match metadata (166 matches)
    ├── teams.parquet                       # National team metadata (48 teams)
    ├── players.parquet                     # Player appearances and minutes played (1,254 players)
    └── player_summary_ratings.parquet      # Aggregated player ratings across xT, VAEP, and Atomic-VAEP
```

---

## Quickstart & Reproduction Instructions

```bash
# 1. Create a Python 3.12 virtual environment and install dependencies
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt

# 2. Run the end-to-end data & modeling pipeline (Optional if precomputed summary files are present)
python src/01_data_etl.py
python src/02_xt_model.py
python src/03_vaep_models.py
python src/04_generate_match_case_study.py

# 3. Launch the interactive Streamlit web application
streamlit run app.py
```
