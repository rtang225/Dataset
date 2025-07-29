import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import time
import datetime

# Read the CSV file and select only the required columns
df = pd.read_csv('originaldata.csv', usecols=['DATE_DEBUT', 'LATITUDE', 'LONGITUDE', 'SUP_HA', 'CAUSE'])

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
apicalls = 0
minute_start = time.time()
hour_start = time.time()

# For each row in the CSV, fetch weather data for the date and location
for index, row in df.iterrows():    
    date = row['DATE_DEBUT']
    lat = row['LATITUDE']
    lon = row['LONGITUDE']
    area = row['SUP_HA']
    cause = row['CAUSE']
    url = "https://customer-archive-api.open-meteo.com/v1/archive"
    try:
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    except Exception as e:
        continue
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date - datetime.timedelta(days=7),
        "end_date": date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "rain", "pressure_msl", "surface_pressure", "cloud_cover", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm", "et0_fao_evapotranspiration", "wind_gusts_10m", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"],
        "apikey": "14P5w4g2sL9XCSX3"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
    hourly_rain = hourly.Variables(5).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(6).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(7).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(8).ValuesAsNumpy()
    hourly_vapour_pressure_deficit = hourly.Variables(9).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(10).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(11).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(12).ValuesAsNumpy()
    hourly_wind_direction_100m = hourly.Variables(13).ValuesAsNumpy()
    hourly_soil_temperature_0_to_7cm = hourly.Variables(14).ValuesAsNumpy()
    hourly_soil_temperature_7_to_28cm = hourly.Variables(15).ValuesAsNumpy()
    hourly_soil_temperature_28_to_100cm = hourly.Variables(16).ValuesAsNumpy()
    hourly_soil_temperature_100_to_255cm = hourly.Variables(17).ValuesAsNumpy()
    hourly_soil_moisture_0_to_7cm = hourly.Variables(18).ValuesAsNumpy()
    hourly_soil_moisture_7_to_28cm = hourly.Variables(19).ValuesAsNumpy()
    hourly_soil_moisture_28_to_100cm = hourly.Variables(20).ValuesAsNumpy()
    hourly_soil_moisture_100_to_255cm = hourly.Variables(21).ValuesAsNumpy()
    hourly_et0_fao_evapotranspiration = hourly.Variables(22).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(23).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(24).ValuesAsNumpy()
    hourly_cloud_cover_mid = hourly.Variables(25).ValuesAsNumpy()
    hourly_cloud_cover_high = hourly.Variables(26).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["date"] = hourly_data["date"].strftime('%Y-%m-%d %H:%M:%S').tolist()
    hourly_data["latitude"] = lat
    hourly_data["longitude"] = lon
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["dew_point_2m"] = hourly_dew_point_2m
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["rain"] = hourly_rain
    hourly_data["pressure_msl"] = hourly_pressure_msl
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
    hourly_data["soil_temperature_0_to_7cm"] = hourly_soil_temperature_0_to_7cm
    hourly_data["soil_temperature_7_to_28cm"] = hourly_soil_temperature_7_to_28cm
    hourly_data["soil_temperature_28_to_100cm"] = hourly_soil_temperature_28_to_100cm
    hourly_data["soil_temperature_100_to_255cm"] = hourly_soil_temperature_100_to_255cm
    hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
    hourly_data["soil_moisture_7_to_28cm"] = hourly_soil_moisture_7_to_28cm
    hourly_data["soil_moisture_28_to_100cm"] = hourly_soil_moisture_28_to_100cm
    hourly_data["soil_moisture_100_to_255cm"] = hourly_soil_moisture_100_to_255cm
    hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
    hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
    hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
    hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
    hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
    hourly_data["area"] = area

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    print(hourly_dataframe)

    # Write or append to CSV
    if cause == "Humaine":
        file_exists = os.path.isfile('human.csv')
        hourly_dataframe.to_csv('human.csv', mode='a', header=not file_exists, index=False)
    if cause == "Foudre":
        file_exists = os.path.isfile('lightning.csv')
        hourly_dataframe.to_csv('lightning.csv', mode='a', header=not file_exists, index=False)