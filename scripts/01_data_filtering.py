

Library used: pandas
"""

import pandas as pd

# -----------------------------------------------------------------
# STEP 1: Bird Migration Dataset -> Filter to United States only
# -----------------------------------------------------------------
# Original file: Bird_migration_dataset.xls (converted to CSV first,
# since it was a global dataset with 42,844 rows across many countries)

migration = pd.read_csv("Bird_migration_dataset.csv")

print("Original migration rows:", len(migration))
print("Countries present:", migration["Countries"].nunique())

# Filter: keep only rows where Country == United States
migration_us = migration[migration["Countries"] == "United States"].copy()
print("After US filter:", len(migration_us))

migration_us.to_csv("bird_migration_US_filtered.csv", index=False)

# Optional stricter version: also match bird-strike years (2000-2011)
migration_us_2000_2011 = migration_us[
    (migration_us["Migration start year"] >= 2000)
    & (migration_us["Migration start year"] <= 2011)
]
print("After US + 2000-2011 filter:", len(migration_us_2000_2011))
migration_us_2000_2011.to_csv("bird_migration_US_2000_2011.csv", index=False)


# -----------------------------------------------------------------
# STEP 2: Check weather station coverage
# -----------------------------------------------------------------
weather = pd.read_csv("historical_weather_dataset_2000_2011.csv")
print("Weather stations found:", weather["Station_ID"].unique())
# Output: ['KJFK_New_York', 'KDEN_Denver', 'KORD_Chicago']
# -> Only 3 stations => only 3 states have matching weather data:
#    New York (NY), Colorado (CO), Illinois (IL)

target_states = ["NY", "IL", "CO"]


# -----------------------------------------------------------------
# STEP 3: Scope bird strikes, airports, and wind turbines
#          to the same 3 states as the weather data
# -----------------------------------------------------------------

# --- Bird strikes ---
bird_strikes = pd.read_csv("cleaned_bird_strikes.csv")
bird_strikes_scoped = bird_strikes[
    bird_strikes["origin_state_abbr"].isin(target_states)
]
print("Bird strikes scoped:", len(bird_strikes_scoped))
bird_strikes_scoped.to_csv("bird_strikes_NY_IL_CO.csv", index=False)

# --- Airports ---
airports = pd.read_csv("cleaned_airports.csv")
airports_scoped = airports[airports["state_abbr"].isin(target_states)]
print("Airports scoped:", len(airports_scoped))
airports_scoped.to_csv("airports_NY_IL_CO.csv", index=False)

# --- Wind turbines ---
turbines = pd.read_csv("cleaned_wind_turbines.csv")
turbines_scoped = turbines[turbines["state_abbr"].isin(target_states)]
print("Wind turbines scoped:", len(turbines_scoped))
turbines_scoped.to_csv("wind_turbines_NY_IL_CO.csv", index=False)


# -----------------------------------------------------------------
# Result: 6 final files, all aligned to New York, Illinois, Colorado
# -----------------------------------------------------------------
# 1. historical_weather_dataset_2000_2011.csv   (unchanged, 3 stations)
# 2. bird_strikes_NY_IL_CO.csv                  (2,987 rows)
# 3. airports_NY_IL_CO.csv                      (2,602 rows)
# 4. wind_turbines_NY_IL_CO.csv                 (7,094 rows)
# 5. bird_migration_US_filtered.csv             (2,372 rows)
# 6. bird_migration_US_2000_2011.csv            (647 rows, optional)
