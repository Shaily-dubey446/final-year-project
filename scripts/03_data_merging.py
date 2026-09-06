

import pandas as pd
import numpy as np
import difflib

# ============================================================
# STEP A: Weather -> daily average per station
# ============================================================
weather = pd.read_csv("historical_weather_dataset_2000_2011_clean.csv")
weather["Timestamp"] = pd.to_datetime(weather["Timestamp"])
weather["date"] = weather["Timestamp"].dt.date

weather_daily = weather.groupby(["Station_ID", "date"]).agg(
    avg_temp_c=("Temperature_C", "mean"),
    avg_humidity_pct=("Humidity_Pct", "mean"),
    avg_wind_speed_kmh=("Wind_Speed_kmh", "mean"),
    avg_visibility_km=("Visibility_km", "mean"),
    total_precip_mm=("Precipitation_mm", "sum"),
    avg_pressure_hpa=("Pressure_hPa", "mean"),
).reset_index()

station_to_state = {
    "KJFK_New_York": "NY",
    "KORD_Chicago": "IL",
    "KDEN_Denver": "CO",
}
weather_daily["state_abbr"] = weather_daily["Station_ID"].map(station_to_state)
weather_daily["date"] = pd.to_datetime(weather_daily["date"])


# ============================================================
# STEP B: Bird strikes + Airports (fuzzy match on airport name)
# ============================================================
bs = pd.read_csv("bird_strikes_NY_IL_CO_clean.csv")
ap = pd.read_csv("airports_NY_IL_CO_clean.csv")

bs["airport_name_clean"] = bs["airport_name"].str.upper().str.strip()
ap["airport_name_clean"] = ap["airport_name"].str.upper().str.strip()

ap_lookup = dict(
    zip(
        ap["airport_name_clean"],
        zip(ap["latitude"], ap["longitude"], ap["elevation_ft"]),
    )
)
ap_names_list = list(ap_lookup.keys())


def find_airport_coords(name):
    """Fuzzy-match a bird-strike airport name to the airports table
    (exact string match fails most of the time because of different
    naming conventions, e.g. 'ARPT' vs 'AIRPORT')."""
    match = difflib.get_close_matches(name, ap_names_list, n=1, cutoff=0.6)
    if match:
        return ap_lookup[match[0]]
    return (np.nan, np.nan, np.nan)


coords = bs["airport_name_clean"].apply(find_airport_coords)
bs["latitude"] = coords.apply(lambda x: x[0])
bs["longitude"] = coords.apply(lambda x: x[1])
bs["elevation_ft"] = coords.apply(lambda x: x[2])


# ============================================================
# STEP C: Nearest wind turbine distance (Haversine formula)
# ============================================================
wt = pd.read_csv("wind_turbines_NY_IL_CO_clean.csv")


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two GPS points."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_turbine_dist(lat, lon, state):
    if pd.isna(lat):
        return np.nan
    subset = wt[wt["state_abbr"] == state]
    if len(subset) == 0:
        return np.nan
    dists = haversine(lat, lon, subset["latitude"].values, subset["longitude"].values)
    return dists.min()


bs["nearest_turbine_km"] = bs.apply(
    lambda r: nearest_turbine_dist(r["latitude"], r["longitude"], r["origin_state_abbr"]),
    axis=1,
)


# ============================================================
# STEP D: Merge bird strikes with daily weather (state + date)
# ============================================================
bs["flight_date"] = pd.to_datetime(bs["flight_date"])
bs["state_abbr"] = bs["origin_state_abbr"]

master = bs.merge(
    weather_daily,
    left_on=["state_abbr", "flight_date"],
    right_on=["state_abbr", "date"],
    how="left",
)


# ============================================================
# STEP E: Migration season flag
# ============================================================
# NOTE (limitation to mention in the report): this simple version
# marks a month "active" if ANY species in the migration dataset
# starts/ends migration in that month. Because different species
# migrate in different months, this currently flags all 12 months
# as active, i.e. the feature isn't discriminative yet. A better
# version would match per-species and per-region before using this
# as a model feature.
mig = pd.read_csv("bird_migration_US_filtered_clean.csv")
active_months = set()
for _, row in mig.iterrows():
    sm, em = row["Migration start month"], row["Migration end month"]
    if pd.notna(sm) and pd.notna(em):
        active_months.add(int(sm))
        active_months.add(int(em))

master["flight_month"] = master["flight_date"].dt.month
master["is_migration_season"] = master["flight_month"].isin(active_months).astype(int)


# ============================================================
# Save master merged dataset
# ============================================================
master.to_csv("master_dataset.csv", index=False)

print("Final master dataset shape:", master.shape)
print("Rows with airport coordinates matched:", master["latitude"].notna().sum())
print("Rows missing airport match (drop or handle before modeling):",
      master["latitude"].isnull().sum())
