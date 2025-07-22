import pandas as pd
from datetime import timedelta
import numpy as np
import time

# Load data
orig_df = pd.read_csv('originaldata.csv')
week_df = pd.read_csv('week.csv')
rows = orig_df.shape[0]

# Ensure datetime format
orig_df['DATE_DEBUT'] = pd.to_datetime(orig_df['DATE_DEBUT'])
week_df['date'] = pd.to_datetime(week_df['date'])

# Group and create sequences
def get_week_sequence(lat, lon, ref_date):
    # Get all rows for this lat/lon in the week before ref_date (not including ref_date)
    mask = (
        (week_df['latitude'] == lat) &
        (week_df['longitude'] == lon) &
        (week_df['date'] < ref_date) &
        (week_df['date'] >= ref_date - timedelta(days=7))
    )
    week_data = week_df[mask].sort_values('date')
    return week_data

start_time = time.time()
# Build sequences for LSTM
sequences = []
targets = []
for idx, row in orig_df.iterrows():
    lat = row['LATITUDE']
    lon = row['LONGITUDE']
    ref_date = row['DATE_DEBUT']
    week_data = get_week_sequence(lat, lon, ref_date)
    if len(week_data) > 0:
        # Use all numeric columns except lat/lon/date as features
        # 'temperature_2m', 'wind_speed_10m', 'wind_speed_100m', 'relative_humidity_2m', 'vapour_pressure_deficit', 'area', 'apparent_temperature'
        # 'rain', 'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'dew_point_2m', 'wind_gusts_10m'
        # 'surface_pressure', 'pressure_msl', 'et0_fao_evaporation', 'soil_moisture_28_to_100cm'
        # feature_cols = [c for c in week_data.columns if c not in ['date', 'latitude', 'longitude', 'area']]
        feature_cols = [c for c in week_data.columns if c in ['temperature_2m', 'wind_speed_10m', 'wind_speed_100m', 'relative_humidity_2m', 'vapour_pressure_deficit', 'apparent_temperature', 'rain', 'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'dew_point_2m', 'wind_gusts_10m', 'surface_pressure', 'pressure_msl', 'et0_fao_evaporation', 'soil_moisture_28_to_100cm']]
        seq = week_data[feature_cols].values
        sequences.append(seq)
        targets.append(row['SUP_HA'] if 'SUP_HA' in row else 0)
    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {idx + 1} out of {rows} rows in {elapsed:.2f} seconds.")

elapsed = time.time() - start_time
print(f"Total time: {elapsed:.2f} seconds.")

# Save sequences and targets for later training
np.save('week_sequences_r3.npy', np.array(sequences, dtype=object), allow_pickle=True)
np.save('week_targets_r3.npy', np.array(targets), allow_pickle=True)

print('Done! Sequences and targets saved.')