# 🚚 Supply Chain Forecasting

**Demand Forecasting & Supplier Risk Analysis using Random Forest**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This project forecasts product demand and quantifies supplier risk across an enterprise supply chain using **Random Forest Regression**. It includes three-scenario analysis (Base / Growth / Risk), feature importance ranking, and strategic operational recommendations.

Supply chain disruptions cost businesses billions annually. This model helps operations and procurement teams anticipate demand, identify high-risk suppliers, and plan inventory proactively.

---

## 🧠 How It Works

```
Raw Supply Chain Data (180 days × 10 products)
      │
      ▼
Feature Engineering ──► Month, Quarter, DayOfWeek, Product encoding
      │
      ▼
Random Forest Regressor ──► Train on 80%, test on 20%
      │
      ▼
Model Evaluation ──► MAE, R² Score
      │
      ▼
Scenario Forecasting ──► Base / Growth / Risk scenarios
      │
      ▼
Risk Analysis + Report + CSV Output
```

---

## 📁 Project Structure

```
supply-chain-forecasting/
├── data/
│   └── supply_chain.csv          # Input data (auto-generated if missing)
├── notebooks/
│   └── forecasting_analysis.ipynb
├── src/
│   └── forecasting.py            # Full forecasting pipeline
├── outputs/
│   └── forecast_results.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Features Used

| Feature | Description |
|---------|-------------|
| `Marketing_Budget` | Spend allocated to product marketing |
| `Inventory_Level` | Current stock on hand |
| `Shipping_Delay_Days` | Days delayed in transit |
| `Supplier_Risk_Score` | Supplier reliability score (1–5) |
| `Seasonality_Index` | Seasonal demand multiplier (1–5) |
| `Promo_Flag` | 1 if promotional event active |
| `Month / Quarter / DayOfWeek` | Time-based demand patterns |
| `Product_Enc` | Label-encoded product category |

---

## 📊 Scenario Analysis

| Scenario | Description |
|----------|-------------|
| **Base Case** | Normal operations, no promotion |
| **Growth Push** | High marketing spend + active promotion |
| **Risk Scenario** | Low budget + high supplier risk |

---

## 🚀 Getting Started

```bash
git clone https://github.com/Fariba6384/supply-chain-forecasting.git
cd supply-chain-forecasting
pip install -r requirements.txt
python src/forecasting.py
```

---

## 👤 Author

**Fariba Seyedjafarrangraz**  
Business Transformation Analyst · Dalhousie University  
CBAP Certified · PhD · Halifax, Nova Scotia, Canada  
📧 rangrazfariba6384@gmail.com · 🐙 [github.com/Fariba6384](https://github.com/Fariba6384)
