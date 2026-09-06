
import pandas as pd

# -----------------------------------------------------------------
# 1. WEATHER DATA — remove unrealistic temperature outliers
# -----------------------------------------------------------------
weather = pd.read_csv("historical_weather_dataset_2000_2011.csv")

print("Duplicate rows in weather:", weather.duplicated().sum())  # 0 found

before = len(weather)
# JFK / ORD / DEN realistically stay within -25C to 45C historically.
# Anything beyond this is treated as a sensor/entry error.
weather = weather[
    (weather["Temperature_C"] >= -25) & (weather["Temperature_C"] <= 45)
]
print(f"Weather: removed {before - len(weather)} outlier rows")

weather.to_csv("historical_weather_dataset_2000_2011_clean.csv", index=False)


# -----------------------------------------------------------------
# 2. BIRD STRIKES — fill missing 'engines' values
# -----------------------------------------------------------------
bird_strikes = pd.read_csv("bird_strikes_NY_IL_CO.csv")

print("Duplicate rows in bird_strikes:", bird_strikes.duplicated().sum())  # 0 found
print("Missing 'engines' values:", bird_strikes["engines"].isnull().sum())

# Fill missing engine counts with the median (most aircraft in this
# dataset have 2 engines, so median is a safe, non-distorting choice)
bird_strikes["engines"] = bird_strikes["engines"].fillna(
    bird_strikes["engines"].median()
)

bird_strikes.to_csv("bird_strikes_NY_IL_CO_clean.csv", index=False)


# -----------------------------------------------------------------
# 3. AIRPORTS — drop closed airports, fill missing elevation/municipality
# -----------------------------------------------------------------
airports = pd.read_csv("airports_NY_IL_CO.csv")

print("Duplicate rows in airports:", airports.duplicated().sum())  # 0 found

before = len(airports)
# Closed airports are not operational -> irrelevant to collision risk
airports = airports[airports["type"] != "closed"]
print(f"Airports: dropped {before - len(airports)} closed airports")

airports["elevation_ft"] = airports["elevation_ft"].fillna(
    airports["elevation_ft"].median()
)
airports["municipality"] = airports["municipality"].fillna("Unknown")

# Note: icao_code / iata_code nulls are NOT filled — small airports/
# heliports genuinely don't have these codes, so it's not "dirty" data,
# it's expected missingness. Left as-is.

airports.to_csv("airports_NY_IL_CO_clean.csv", index=False)


# -----------------------------------------------------------------
# 4. WIND TURBINES — checked, already clean (no nulls, no duplicates,
#    no zero/negative capacities) -> saved as-is
# -----------------------------------------------------------------
turbines = pd.read_csv("wind_turbines_NY_IL_CO.csv")
print("Duplicate rows in turbines:", turbines.duplicated().sum())  # 0 found
turbines.to_csv("wind_turbines_NY_IL_CO_clean.csv", index=False)


# -----------------------------------------------------------------
# 5. MIGRATION DATA — checked, already clean -> saved as-is
# -----------------------------------------------------------------
migration = pd.read_csv("bird_migration_US_filtered.csv")
print("Duplicate rows in migration:", migration.duplicated().sum())  # 0 found
migration.to_csv("bird_migration_US_filtered_clean.csv", index=False)


# -----------------------------------------------------------------
# Summary of what was cleaned
# -----------------------------------------------------------------
# - Weather: 17 outlier rows removed (unrealistic temperatures)
# - Bird strikes: 11 missing 'engines' values filled with median
# - Airports: 627 closed airports dropped, elevation/municipality
#             nulls filled
# - Turbines: no issues found
# - Migration: no issues found
