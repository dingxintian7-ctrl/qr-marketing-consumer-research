"""Generate synthetic questionnaire data for a reproducible methods demonstration.

The generated observations are NOT the original project's questionnaire data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
N = 300

root = Path(__file__).resolve().parents[1]
out = root / "data" / "demo_qr_marketing_survey.csv"

age = RNG.integers(18, 46, N)
shopping_frequency = RNG.integers(1, 6, N)
incentive_attractiveness = RNG.integers(1, 6, N)
content_usefulness = RNG.integers(1, 6, N)
ease_of_use = RNG.integers(1, 6, N)
scenario_fit = RNG.integers(1, 6, N)

noise = RNG.normal(0, 0.65, N)
marketing_effect = (
    0.24 * incentive_attractiveness
    + 0.27 * content_usefulness
    + 0.22 * ease_of_use
    + 0.27 * scenario_fit
    + noise
)
marketing_effect = np.clip(np.rint(marketing_effect), 1, 5).astype(int)

df = pd.DataFrame({
    "respondent_id": np.arange(1, N + 1),
    "age": age,
    "shopping_frequency": shopping_frequency,
    "incentive_attractiveness": incentive_attractiveness,
    "content_usefulness": content_usefulness,
    "ease_of_use": ease_of_use,
    "scenario_fit": scenario_fit,
    "marketing_effect_score": marketing_effect,
})

# Insert a few missing values to demonstrate quality checks.
for col in ["content_usefulness", "ease_of_use", "scenario_fit"]:
    idx = RNG.choice(df.index, size=4, replace=False)
    df.loc[idx, col] = np.nan

out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Demo data written to: {out}")
