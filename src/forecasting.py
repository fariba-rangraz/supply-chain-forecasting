"""
forecasting.py
==============
Supply Chain Demand Forecasting & Supplier Risk Analysis

Uses Random Forest Regression to forecast product demand across multiple
scenarios and quantifies supplier risk — supporting inventory planning,
logistics optimisation, and procurement decisions.

Author : Fariba Seyedjafarrangraz
         Business Transformation Analyst | CBAP | PhD
         Dalhousie University — Registrar's Office
         Halifax, Nova Scotia, Canada
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


# ── CONFIG ────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
DATA_PATH    = Path(__file__).parent.parent / "data" / "supply_chain.csv"
OUTPUT_PATH  = Path(__file__).parent.parent / "outputs" / "forecast_results.csv"

PRODUCTS = [
    "Laptop", "Phone", "Tablet", "Monitor", "Keyboard",
    "Mouse", "Printer", "Camera", "Speaker", "Smartwatch",
]

FEATURES = [
    "Marketing_Budget", "Inventory_Level", "Shipping_Delay_Days",
    "Supplier_Risk_Score", "Seasonality_Index", "Promo_Flag",
    "Month", "Quarter", "DayOfWeek", "Product_Enc",
]

SCENARIOS = {
    "Base Case":     {"Marketing_Budget": 3000, "Promo_Flag": 0, "Supplier_Risk_Score": 3},
    "Growth Push":   {"Marketing_Budget": 4800, "Promo_Flag": 1, "Supplier_Risk_Score": 2},
    "Risk Scenario": {"Marketing_Budget": 2000, "Promo_Flag": 0, "Supplier_Risk_Score": 5},
}


# ── 1. DATA ───────────────────────────────────────────────────────────────────

def load_or_generate(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"])

    print("ℹ  No data file found — generating synthetic supply chain dataset.")
    np.random.seed(RANDOM_STATE)
    dates   = pd.date_range("2024-01-01", periods=180, freq="D")
    records = []

    for date in dates:
        for product in PRODUCTS:
            marketing      = np.random.randint(1_000, 5_000)
            inventory      = np.random.randint(50, 500)
            shipping_delay = np.random.randint(0, 10)
            supplier_risk  = np.random.randint(1, 6)
            seasonality    = np.random.randint(1, 5)
            promo_flag     = int(np.random.rand() > 0.80)

            demand = max(0, round(
                marketing      * 0.80
                + inventory    * 1.50
                - shipping_delay * 120
                - supplier_risk  * 200
                + seasonality    * 300
                + promo_flag     * 800
                + np.random.normal(0, 500)
            ))

            records.append({
                "Date": date, "Product": product,
                "Marketing_Budget": marketing, "Inventory_Level": inventory,
                "Shipping_Delay_Days": shipping_delay, "Supplier_Risk_Score": supplier_risk,
                "Seasonality_Index": seasonality, "Promo_Flag": promo_flag, "Demand": demand,
            })

    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✔  Dataset saved → {path}\n")
    return df


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    df["Month"]     = pd.to_datetime(df["Date"]).dt.month
    df["Quarter"]   = pd.to_datetime(df["Date"]).dt.quarter
    df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.dayofweek
    le              = LabelEncoder()
    df["Product_Enc"] = le.fit_transform(df["Product"])
    return df, le


# ── 3. MODEL ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["Demand"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, X, {
        "MAE": mean_absolute_error(y_test, preds),
        "R2":  r2_score(y_test, preds),
    }


# ── 4. SCENARIO FORECASTING ───────────────────────────────────────────────────

SCENARIO_BASE = {
    "Inventory_Level": 320, "Shipping_Delay_Days": 2,
    "Seasonality_Index": 4, "Month": 9,
    "Quarter": 3, "DayOfWeek": 2, "Product_Enc": 0,
}

def forecast_scenarios(model, feature_cols: list) -> dict:
    results = {}
    for name, overrides in SCENARIOS.items():
        row = pd.DataFrame([{**SCENARIO_BASE, **overrides}])[feature_cols]
        results[name] = model.predict(row)[0]
    return results


# ── 5. RISK ANALYSIS ──────────────────────────────────────────────────────────

def risk_summary(df: pd.DataFrame) -> dict:
    return {
        "avg_delay":       df["Shipping_Delay_Days"].mean(),
        "high_risk_count": (df["Supplier_Risk_Score"] >= 4).sum(),
        "high_risk_pct":   (df["Supplier_Risk_Score"] >= 4).mean() * 100,
        "top_products":    df.groupby("Product")["Demand"].mean().sort_values(ascending=False),
    }


# ── 6. REPORT ─────────────────────────────────────────────────────────────────

def print_report(df, metrics, scenarios, risk, model, feature_cols):
    DIV = "─" * 70

    def section(t):
        print(f"\n▌ {t}\n{DIV}")

    print("\n" + "═" * 70)
    print("   SUPPLY CHAIN FORECASTING REPORT")
    print("   Demand Forecasting & Supplier Risk Analytics")
    print("═" * 70)

    section("Dataset Overview")
    print(f"  Records    : {len(df):,}")
    print(f"  Products   : {df['Product'].nunique()}")
    print(f"  Date Range : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Avg Demand : {df['Demand'].mean():,.0f} units/record")

    section("Model Performance")
    print(f"  Algorithm  : Random Forest Regressor (n=300, depth=12)")
    print(f"  MAE        : {metrics['MAE']:,.0f} units")
    print(f"  R² Score   : {metrics['R2']:.4f}")

    section("Scenario Demand Forecasts")
    for name, val in scenarios.items():
        bar = "█" * int(val / 400)
        print(f"  {name:<18}: {val:>8,.0f} units  {bar}")

    section("Supplier Risk KPIs")
    print(f"  Avg Shipping Delay     : {risk['avg_delay']:.2f} days")
    print(f"  High-Risk Rows (≥4)    : {risk['high_risk_count']:,} ({risk['high_risk_pct']:.1f}%)")

    section("Top 5 Products by Avg Demand")
    for i, (prod, demand) in enumerate(risk["top_products"].head(5).items(), 1):
        print(f"  {i}. {prod:<14} : {demand:,.0f} units")

    section("Top Predictive Features")
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, val in imp.head(6).items():
        bar = "█" * int(val * 60)
        print(f"  {feat:<26} {val:.4f}  {bar}")

    section("Strategic Insights")
    if risk["avg_delay"] > 5:
        print("  ⚠  Avg delay >5 days — review logistics contracts.")
    if risk["high_risk_pct"] > 20:
        print("  ⚠  >20% supplier risk exposure — diversify procurement.")
    base, growth = scenarios["Base Case"], scenarios["Growth Push"]
    if growth > base * 1.25:
        print(f"  ✔  Promotions deliver {(growth/base - 1)*100:.0f}% uplift — scale budget.")
    if metrics["R2"] > 0.85:
        print("  ✔  R²>0.85 — model is production-ready.")
    top = risk["top_products"].index[0]
    print(f"  ◉  Prioritise inventory for '{top}' — highest avg demand.")

    print(f"\n{DIV}\n  Report complete.\n{DIV}\n")


# ── 7. SAVE ───────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✔  Results saved → {OUTPUT_PATH}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, le          = engineer_features(load_or_generate(DATA_PATH))
    model, X, metrics = train(df)
    scenarios       = forecast_scenarios(model, FEATURES)
    risk            = risk_summary(df)
    print_report(df, metrics, scenarios, risk, model, FEATURES)
    save(df)
