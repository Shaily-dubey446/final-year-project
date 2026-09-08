"""
Bird Collision Risk Prediction - v2: Adds a 'season' feature
------------------------------------------------------------------
This is a SEPARATE version of the feature-engineering step. It does
NOT replace the original pipeline (00-04 scripts) or the original
bird_collision_model_dataset_FINAL.csv — it produces a new file
alongside it, so both versions (with/without season) can be trained
and compared.

Input : master_dataset.csv (output of 03_data_merging.py)
Output: bird_collision_model_dataset_v2_with_season.csv

The only change from the original pipeline (04_feature_engineering_final.py)
is the addition of a 'season' column, derived from flight_month:
    Dec, Jan, Feb -> Winter
    Mar, Apr, May -> Spring
    Jun, Jul, Aug -> Summer
    Sep, Oct, Nov -> Fall
"""

import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Load merged dataset
# ============================================================
df = pd.read_csv("master_dataset.csv", low_memory=False)
print("Loaded:", df.shape)

# ============================================================
# STEP 2: Remove duplicate rows (if any)
# ============================================================
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

# ============================================================
# STEP 3: Fix missing airport coordinates (same as original pipeline)
# ============================================================
manual_map = {
    "ALBANY INTL":            (42.748299, -73.801697, 285.0),
    "GREATER ROCKFORD":       (42.1954,   -89.097198, 742.0),
    "GREATER ROCHESTER INTL": (43.1189,   -77.672401, 559.0),
    "BLOOMINGTON/NORMAL":     (40.4771,   -88.915901, 871.0),
    "FORT COLLINS-LOVELAND":  (40.41207, -105.11256,  5080.0),
    "GUNNISON COUNTY":        (38.534672,-106.934566, 7680.0),
    "MOUNT HAWLEY":           (40.7953,  -89.613403,  786.0),
}
for name, (lat, lon, elev) in manual_map.items():
    mask = df["airport_name_clean"] == name
    df.loc[mask, "latitude"] = lat
    df.loc[mask, "longitude"] = lon
    df.loc[mask, "elevation_ft"] = elev

wt = pd.read_csv("cleaned_wind_turbines.csv")
wt = wt[wt["state_abbr"].isin(["NY", "IL", "CO"])]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_dist(lat, lon, state):
    subset = wt[wt["state_abbr"] == state]
    return haversine(lat, lon, subset["latitude"].values, subset["longitude"].values).min()


still_missing = df["nearest_turbine_km"].isnull()
df.loc[still_missing, "nearest_turbine_km"] = df[still_missing].apply(
    lambda r: nearest_dist(r["latitude"], r["longitude"], r["state_abbr"]), axis=1
)
print("Total nulls remaining:", df.isnull().sum().sum())

# ============================================================
# STEP 4 (NEW): Add the 'season' column
# ============================================================
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


df["season"] = df["flight_month"].apply(get_season)
print("\nSeason distribution:")
print(df["season"].value_counts())

# ============================================================
# STEP 5: Engineer the target variable (same formula as original)
# ============================================================
def qscore(s):
    q1, q2 = s.quantile([1 / 3, 2 / 3])
    return np.select([s <= q1, s <= q2], [0, 1], default=2)


count = pd.to_numeric(df["number_struck_actual"], errors="coerce").fillna(0)
count_score = qscore(count)

size_map = {"Small": 0, "Medium": 1, "Large": 2}
size_score = df["wildlife_size"].map(size_map).fillna(1)

migration_score = pd.to_numeric(df["is_migration_season"], errors="coerce").fillna(0).clip(0, 1) * 2

alt = pd.to_numeric(df["altitude"], errors="coerce")
alt_q1, alt_q2 = alt.quantile(1 / 3), alt.quantile(2 / 3)
alt_score = pd.Series(
    np.select([alt <= alt_q1, alt <= alt_q2], [2, 1], default=0), index=df.index
).fillna(1)

phase_map = {
    "Takeoff": 2, "Climb": 2, "Approach": 2, "Landing Roll": 2, "Landing": 2,
    "Descent": 1, "En Route": 0, "Taxi": 0, "Parked": 0,
}
phase_score = df["flight_phase"].map(phase_map).fillna(1)

wind = pd.to_numeric(df["avg_wind_speed_kmh"], errors="coerce")
wind_score = pd.Series(qscore(wind), index=df.index)

vis = pd.to_numeric(df["avg_visibility_km"], errors="coerce")
vq1, vq2 = vis.quantile([1 / 3, 2 / 3])
visibility_score = pd.Series(
    np.select([vis <= vq1, vis <= vq2], [2, 1], default=0), index=df.index
).fillna(1)

precip = pd.to_numeric(df["total_precip_mm"], errors="coerce").fillna(0)
precip_score = (precip > 0).astype(int)

dist = pd.to_numeric(df["nearest_turbine_km"], errors="coerce")
dq1, dq2 = dist.quantile([1 / 3, 2 / 3])
turbine_score = pd.Series(
    np.select([dist <= dq1, dist <= dq2], [2, 1], default=0), index=df.index
).fillna(1)

risk_index = (
    0.20 * count_score + 0.10 * size_score + 0.15 * migration_score
    + 0.15 * alt_score + 0.10 * phase_score + 0.08 * wind_score
    + 0.07 * visibility_score + 0.05 * precip_score + 0.10 * turbine_score
)

r1, r2 = risk_index.quantile([1 / 3, 2 / 3])
df["risk_index"] = risk_index.round(4)
df["risk_level"] = np.select([risk_index <= r1, risk_index <= r2], ["Low", "Medium"], default="High")
df["risk_target"] = df["risk_level"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)

print("\nrisk_level distribution:")
print(df["risk_level"].value_counts())

# ============================================================
# STEP 6: Rename misleading column (same as original)
# ============================================================
df = df.rename(columns={"is_aircraft_large": "is_high_altitude"})

# Save full reference version (includes risk_index)
df.to_csv("master_dataset_v2_season_full_reference.csv", index=False)

# ============================================================
# STEP 7: Drop redundant / post-event / leakage columns
# ============================================================
remove_cols = [
    "record_id", "make_model", "operator", "remains_collected",
    "remains_sent_to_smithsonian", "remarks", "cost", "people_injured",
    "damage", "effect", "origin_country", "origin_state_abbr",
    "airport_name", "date", "conditions_precipitation", "risk_index",
]
remove_cols = [c for c in remove_cols if c in df.columns]
df_model = df.drop(columns=remove_cols)

# Re-check for duplicates that only appear after dropping identifier columns
before_final = len(df_model)
df_model = df_model.drop_duplicates()
print(f"\nRemoved {before_final - len(df_model)} rows that became duplicates "
      f"after dropping identifier/outcome columns")

df_model.to_csv("bird_collision_model_dataset_v2_with_season.csv", index=False)

print("\nFull reference file shape:", df.shape)
print("Modeling-ready v2 file shape:", df_model.shape)
print("\nDone. Use 'bird_collision_model_dataset_v2_with_season.csv' to")
print("train v2 models and compare against the original results.")
