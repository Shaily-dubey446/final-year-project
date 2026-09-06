"""
Bird Collision Risk Prediction - Raw Data Cleaning (Step 0)
------------------------------------------------------------------
Cleans and standardizes the 3 raw datasets that our project actually
uses:
  1. Bird_strikes.csv   (FAA-style wildlife strike records)
  2. airports.csv        (global airport reference table -> filtered to US)
  3. wind_turbines.csv   (US wind turbine locations)

NOTE: Weather data is NOT cleaned here. The weather file used later
in this project (historical_weather_dataset_2000_2011.csv) was a
separate, already-scoped 3-station dataset — it did not go through
this script.

Each cleaned file is saved separately to /mnt/user-data/outputs/.
Further filtering/scoping/merging happens in later scripts.
"""

import pandas as pd
import numpy as np
import re

IN_DIR = "/mnt/user-data/uploads"
OUT_DIR = "/mnt/user-data/outputs"

# ------------------------------------------------------------------
# Reference: US state full name -> 2-letter abbreviation
# ------------------------------------------------------------------
US_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
    "Puerto Rico": "PR", "Guam": "GU", "Virgin Islands": "VI",
}


def to_snake_case(col: str) -> str:
    """Convert a column name to clean snake_case, stripping special chars."""
    col = col.replace("?", "")
    col = re.sub(r"(?<!^)(?=[A-Z])", "_", col)   # CamelCase -> Camel_Case
    col = re.sub(r"[.\s]+", "_", col)             # dots/spaces -> underscore
    col = re.sub(r"_+", "_", col)                 # collapse multiple underscores
    return col.strip("_").lower()


# ====================================================================
# 1. BIRD STRIKES
# ====================================================================
def clean_bird_strikes():
    df = pd.read_csv(f"{IN_DIR}/Bird_strikes.csv", low_memory=False)

    # Standardize column names
    df.columns = [to_snake_case(c) for c in df.columns]

    # Parse date (format like "11/23/00 0:00")
    df["flight_date"] = pd.to_datetime(
        df["flight_date"], format="%m/%d/%y %H:%M", errors="coerce"
    )
    df["flight_year"] = df["flight_date"].dt.year
    df["flight_month"] = df["flight_date"].dt.month

    # Cost: remove commas, convert to numeric
    df["cost"] = df["cost"].astype(str).str.replace(",", "", regex=False)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

    # Yes/No -> boolean
    df["pilot_warned"] = df["pilot_warned"].map({"Y": True, "N": False})
    df["is_aircraft_large"] = df["is_aircraft_large"].map({"Yes": True, "No": False})

    # Engines -> numeric
    df["engines"] = pd.to_numeric(df["engines"], errors="coerce")

    # State: standardize, add abbreviation
    df["origin_state"] = df["origin_state"].str.strip()
    df["origin_state_abbr"] = df["origin_state"].map(US_STATE_ABBR)

    # Fill categorical missing values with explicit label (keeps the "absence" info)
    for col in ["effect", "conditions_precipitation", "remarks"]:
        df[col] = df[col].fillna("Not Specified")

    # Deduplicate
    df = df.drop_duplicates()

    # Drop rows with completely invalid dates (can't be used in time-based joins)
    df = df.dropna(subset=["flight_date"])
    df = df.reset_index(drop=True)
    return df


# ====================================================================
# 2. AIRPORTS
# ====================================================================
def clean_airports():
    df = pd.read_csv(f"{IN_DIR}/airports.csv", low_memory=False)

    # Filter to US only (bird strikes / wind turbines / weather are all US)
    df = df[df["iso_country"] == "US"].copy()

    # Keep only columns useful for this project
    keep_cols = [
        "id", "ident", "type", "name", "latitude_deg", "longitude_deg",
        "elevation_ft", "iso_region", "municipality", "scheduled_service",
        "icao_code", "iata_code",
    ]
    df = df[keep_cols]

    # Extract 2-letter state code from iso_region (e.g. "US-PA" -> "PA")
    df["state_abbr"] = df["iso_region"].str.replace("US-", "", regex=False)

    # Rename for clarity/consistency
    df = df.rename(columns={
        "latitude_deg": "latitude",
        "longitude_deg": "longitude",
        "name": "airport_name",
    })

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


# ====================================================================
# 3. WIND TURBINES
# ====================================================================
def clean_wind_turbines():
    df = pd.read_csv(f"{IN_DIR}/wind_turbines.csv", low_memory=False)
    df.columns = [to_snake_case(c) for c in df.columns]

    # NOTE: Site.Latitude / Site.Longitude are swapped in the source file
    # (Site.Latitude values fall in -180..180 range = actual longitude, and
    # vice versa). We fix that here.
    df = df.rename(columns={
        "site_latitude": "longitude_raw",
        "site_longitude": "latitude_raw",
    })
    df["latitude"] = df["latitude_raw"]
    df["longitude"] = df["longitude_raw"]
    df = df.drop(columns=["latitude_raw", "longitude_raw"])

    # Standardize state column name/case
    df["site_state"] = df["site_state"].str.strip().str.upper()
    df = df.rename(columns={"site_state": "state_abbr", "site_county": "county"})

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


# ====================================================================
# RUN PIPELINE
# ====================================================================
if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    bird_strikes = clean_bird_strikes()
    airports = clean_airports()
    wind_turbines = clean_wind_turbines()

    bird_strikes.to_csv(f"{OUT_DIR}/cleaned_bird_strikes.csv", index=False)
    airports.to_csv(f"{OUT_DIR}/cleaned_airports.csv", index=False)
    wind_turbines.to_csv(f"{OUT_DIR}/cleaned_wind_turbines.csv", index=False)

    print("Bird strikes  :", bird_strikes.shape)
    print("Airports (US) :", airports.shape)
    print("Wind turbines :", wind_turbines.shape)
    print("\nAll cleaned files saved to:", OUT_DIR)
