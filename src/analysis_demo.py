"""Reproducible survey-data analysis on synthetic data (not original results)."""
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo_qr_marketing_survey.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Demo dataset not found. Run: python src/generate_demo_data.py")

df = pd.read_csv(DATA)

# 1. Validate column presence, ID uniqueness, and simple value ranges.
required = [
    "respondent_id", "age", "shopping_frequency",
    "incentive_attractiveness", "content_usefulness",
    "ease_of_use", "scenario_fit", "marketing_effect_score"
]
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")
if df["respondent_id"].isna().any() or df["respondent_id"].duplicated().any():
    raise ValueError("Respondent ID is missing or duplicated")
if (~df["age"].dropna().between(18, 100)).any():
    raise ValueError("Age outside the expected range")

score_cols = [
    "shopping_frequency", "incentive_attractiveness", "content_usefulness",
    "ease_of_use", "scenario_fit", "marketing_effect_score"
]
for col in score_cols:
    if (~df[col].dropna().between(1, 5)).any():
        raise ValueError(f"Out-of-range values found in {col}")

# 2. Median imputation is for demonstration only; real studies require a justified method.
clean = df.copy()
for col in score_cols:
    if clean[col].isna().all():
        raise ValueError(f"Cannot impute all-missing column: {col}")
    clean[col] = clean[col].fillna(clean[col].median())

# 3. Descriptive statistics and pairwise correlation.
clean[required].describe().T.to_csv(
    OUT / "descriptive_statistics.csv", encoding="utf-8-sig"
)

analysis_cols = [
    "incentive_attractiveness", "content_usefulness",
    "ease_of_use", "scenario_fit", "marketing_effect_score"
]
clean[analysis_cols].corr().to_csv(
    OUT / "correlation_matrix.csv", encoding="utf-8-sig"
)

# 4. OLS demonstration. Synthetic coefficients have NO substantive interpretation.
X = sm.add_constant(clean[[
    "incentive_attractiveness", "content_usefulness",
    "ease_of_use", "scenario_fit"
]])
y = clean["marketing_effect_score"]
model = sm.OLS(y, X).fit()

with (OUT / "regression_summary.txt").open("w", encoding="utf-8") as f:
    f.write("SYNTHETIC DATA ONLY — these are not original project findings.\n\n")
    f.write(model.summary().as_text())

print("Analysis complete. Results saved to outputs/.")
