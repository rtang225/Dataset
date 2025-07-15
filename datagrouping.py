import pandas as pd
from datetime import timedelta
import numpy as np
import time

# Load data
orig_df = pd.read_csv('fp-historical-wildfire-data-cleaned.csv')
week_df = pd.read_csv('secondset.csv')
rows = orig_df.shape[0]

# Ensure datetime format
orig_df['FIRE_START_DATE'] = pd.to_datetime(orig_df['FIRE_START_DATE'])
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
    ref_date = row['FIRE_START_DATE']
    week_data = get_week_sequence(lat, lon, ref_date)
    if len(week_data) > 0:
        # Use all numeric columns except lat/lon/date as features
        feature_cols = [c for c in week_data.columns if c not in ['date', 'latitude', 'longitude', 'area']]
        seq = week_data[feature_cols].values
        sequences.append(seq)
        targets.append(row['CURRENT_SIZE'] if 'CURRENT_SIZE' in row else 0)
    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {idx + 1} out of {rows} rows in {elapsed:.2f} seconds.")

elapsed = time.time() - start_time
print(f"Total time: {elapsed:.2f} seconds.")

# Save sequences and targets for later training
np.save('week_sequences2.npy', np.array(sequences, dtype=object), allow_pickle=True)
np.save('week_targets2.npy', np.array(targets), allow_pickle=True)

print('Done! Sequences and targets saved.')