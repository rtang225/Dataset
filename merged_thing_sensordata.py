import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import time

# Read the CSV file and select only the required columns
df = pd.read_csv('wf.csv')

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
apicalls = 0
minute_start = time.time()
hour_start = time.time()

# For each row in the CSV, fetch weather data for the date and location
for index, row in df.iterrows():    
    date = row['Date']
    lat = row['Latitude']
    lon = row['Longtitude']
    vNDVI = row['vNDVI']
    VARI = row['VARI']
    area = row['Area']
    url = "https://customer-archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": ["temperature_2m_mean", "apparent_temperature_mean", "precipitation_sum", "rain_sum", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration", "relative_humidity_2m_mean", "pressure_msl_mean", "surface_pressure_mean", "wind_gusts_10m_mean", "wind_speed_10m_mean", "soil_moisture_0_to_100cm_mean", "soil_moisture_0_to_7cm_mean", "soil_moisture_28_to_100cm_mean", "soil_moisture_7_to_28cm_mean", "soil_temperature_0_to_100cm_mean", "soil_temperature_0_to_7cm_mean", "soil_temperature_28_to_100cm_mean", "soil_temperature_7_to_28cm_mean", "vapour_pressure_deficit_max", "leaf_wetness_probability_mean", "dew_point_2m_mean", "et0_fao_evapotranspiration_sum", "cloud_cover_mean", "growing_degree_days_base_0_limit_50", "wet_bulb_temperature_2m_mean", "winddirection_10m_dominant"],
        "apikey": "14P5w4g2sL9XCSX3"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_apparent_temperature_mean = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(3).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(4).ValuesAsNumpy()
    daily_wind_gusts_10m_max = daily.Variables(5).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(6).ValuesAsNumpy()
    daily_shortwave_radiation_sum = daily.Variables(7).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration = daily.Variables(8).ValuesAsNumpy()
    daily_relative_humidity_2m_mean = daily.Variables(9).ValuesAsNumpy()
    daily_pressure_msl_mean = daily.Variables(10).ValuesAsNumpy()
    daily_surface_pressure_mean = daily.Variables(11).ValuesAsNumpy()
    daily_wind_gusts_10m_mean = daily.Variables(12).ValuesAsNumpy()
    daily_wind_speed_10m_mean = daily.Variables(13).ValuesAsNumpy()
    daily_soil_moisture_0_to_100cm_mean = daily.Variables(14).ValuesAsNumpy()
    daily_soil_moisture_0_to_7cm_mean = daily.Variables(15).ValuesAsNumpy()
    daily_soil_moisture_28_to_100cm_mean = daily.Variables(16).ValuesAsNumpy()
    daily_soil_moisture_7_to_28cm_mean = daily.Variables(17).ValuesAsNumpy()
    daily_soil_temperature_0_to_100cm_mean = daily.Variables(18).ValuesAsNumpy()
    daily_soil_temperature_0_to_7cm_mean = daily.Variables(19).ValuesAsNumpy()
    daily_soil_temperature_28_to_100cm_mean = daily.Variables(20).ValuesAsNumpy()
    daily_soil_temperature_7_to_28cm_mean = daily.Variables(21).ValuesAsNumpy()
    daily_vapour_pressure_deficit_max = daily.Variables(22).ValuesAsNumpy()
    daily_leaf_wetness_probability_mean = daily.Variables(23).ValuesAsNumpy()
    daily_dew_point_2m_mean = daily.Variables(24).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration_sum = daily.Variables(25).ValuesAsNumpy()
    daily_cloud_cover_mean = daily.Variables(26).ValuesAsNumpy()
    daily_growing_degree_days_base_0_limit_50 = daily.Variables(27).ValuesAsNumpy()
    daily_wet_bulb_temperature_2m_mean = daily.Variables(28).ValuesAsNumpy()
    daily_winddirection_10m_dominant = daily.Variables(29).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["date"] = daily_data["date"].strftime('%Y-%m-%d').tolist()
    daily_data["latitude"] = lat
    daily_data["longitude"] = lon
    daily_data["vNDVI"] = vNDVI
    daily_data["VARI"] = VARI
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
    daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
    daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration
    daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
    daily_data["pressure_msl_mean"] = daily_pressure_msl_mean
    daily_data["surface_pressure_mean"] = daily_surface_pressure_mean
    daily_data["wind_gusts_10m_mean"] = daily_wind_gusts_10m_mean
    daily_data["wind_speed_10m_mean"] = daily_wind_speed_10m_mean
    daily_data["soil_moisture_0_to_100cm_mean"] = daily_soil_moisture_0_to_100cm_mean
    daily_data["soil_moisture_0_to_7cm_mean"] = daily_soil_moisture_0_to_7cm_mean
    daily_data["soil_moisture_28_to_100cm_mean"] = daily_soil_moisture_28_to_100cm_mean
    daily_data["soil_moisture_7_to_28cm_mean"] = daily_soil_moisture_7_to_28cm_mean
    daily_data["soil_temperature_0_to_100cm_mean"] = daily_soil_temperature_0_to_100cm_mean
    daily_data["soil_temperature_0_to_7cm_mean"] = daily_soil_temperature_0_to_7cm_mean
    daily_data["soil_temperature_28_to_100cm_mean"] = daily_soil_temperature_28_to_100cm_mean
    daily_data["soil_temperature_7_to_28cm_mean"] = daily_soil_temperature_7_to_28cm_mean
    daily_data["vapour_pressure_deficit_max"] = daily_vapour_pressure_deficit_max
    daily_data["leaf_wetness_probability_mean"] = daily_leaf_wetness_probability_mean
    daily_data["dew_point_2m_mean"] = daily_dew_point_2m_mean
    daily_data["et0_fao_evapotranspiration_sum"] = daily_et0_fao_evapotranspiration_sum
    daily_data["cloud_cover_mean"] = daily_cloud_cover_mean
    daily_data["growing_degree_days_base_0_limit_50"] = daily_growing_degree_days_base_0_limit_50
    daily_data["wet_bulb_temperature_2m_mean"] = daily_wet_bulb_temperature_2m_mean
    daily_data["winddirection_10m_dominant"] = daily_winddirection_10m_dominant
    daily_data["area"] = area

    daily_dataframe = pd.DataFrame(data = daily_data)
    apicalls += 3

    # Write or append to CSV
    file_exists = os.path.isfile('export.csv')
    daily_dataframe.to_csv('export.csv', mode='a', header=not file_exists, index=False)