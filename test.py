import pandas as pd
import datetime
df = pd.read_csv('fp-historical-wildfire-data-cleaned.csv')
delete_rows = []
for index, row in df.iterrows():    
    date = row['FIRE_START_DATE']
    try:
        date = datetime.datetime.strptime(date, '%m/%d/%Y %H:%M').date()
    except Exception as e:
        print(index)
        delete_rows.append(index)
"""# Delete the specified rows
# print(delete_rows)
delete_rows.reverse()
df = df.drop(delete_rows)
# Optionally, save the cleaned DataFrame
df.to_csv('fp-historical-wildfire-data-cleaned.csv', index=False)
# print(df.head())"""