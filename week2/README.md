# Week 2 Assignment — Tesla EDA & ML Pipeline

This notebook is my Week 2 assignment submission. The task was to build an end-to-end ML pipeline on sales/price data covering preprocessing, EDA, feature engineering, regression modeling, hyperparameter tuning, and time series forecasting.



## Dataset

**Tesla EA Deliveries and Production Data (2015–2025)** from Kaggle — 2,640 rows where each row represents a unique Year × Month × Region × Model combination. Columns include production units, estimated deliveries, average price, battery capacity, range, CO₂ savings, and charging station counts.

One thing worth noting upfront: a large portion of the data is interpolated or regionally estimated (the `Source_Type` column makes this clear). So a lot of the uniformity you'll see in the EDA — equal regional splits, equal model shares — is a property of how the data was generated, not actual Tesla business patterns. I've called this out in the insights throughout.


## What's in the Notebook

**Sections:**
1. Imports & Setup
2. Data Loading & Overview
3. Data Preprocessing & Aggregation — built 5 aggregated views (overall quarterly, per region, per model, monthly per region, monthly per model)
4. EDA — 14 charts covering production/delivery trends, regional and model breakdowns, pricing, CO₂, seasonality heatmaps, delivery rates, and QoQ growth
5. Feature Engineering — label encoding, cyclical month encoding, temporal features; target is `Avg_Price_USD`
6. Regression Modeling — compared Linear Regression, Ridge, Random Forest, and Gradient Boosting on price prediction
7. Hyperparameter Tuning — GridSearchCV with TimeSeriesSplit on the best model
8. Time Series Forecasting — Holt-Winters and SARIMA on quarterly delivery volume with 8-quarter future projections
9. Summary & Key Findings



## A Few Design Decisions I Want to Highlight

**Target variable:** I chose `Avg_Price_USD` as the regression target since the goal was a sales/price pipeline. The dataset also has delivery volume but using that as a target had serious data leakage issues — `Production_Units` correlates at r=0.994 with deliveries, making any model on it essentially pointless.

**Train/test split:** Used a temporal split (train on 2015–2022, test on 2023–2025) instead of random splitting. Random splitting on time-series structured data leaks future rows into training.

**Cross-validation:** Used `TimeSeriesSplit` instead of standard k-fold for the same reason.

**Features:** Only genuine vehicle attributes and temporal features — `Model`, `Region`, `Battery_Capacity_kWh`, `Range_km`, `Year`, `Quarter_Num`, `Charging_Stations`. No derived or proxy columns that touch the target.


## How to Run

1. Clone the repo or download the notebook
2. Install dependencies:
```
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels kagglehub
```
3. The notebook pulls the dataset directly via `kagglehub` — you'll need a Kaggle account and API token set up, or you can swap the loading cell with:
```python
df_raw = pd.read_csv("tesla_deliveries_dataset_2015_2025.csv")
```
4. Run all cells top to bottom


## Libraries Used

- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — regression models, preprocessing, evaluation
- `statsmodels` — time series (Holt-Winters, SARIMA, ADF test)
- `kagglehub` — dataset loading


