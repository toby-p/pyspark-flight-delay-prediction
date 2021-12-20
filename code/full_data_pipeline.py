"""Full data pipeline including the following steps:
1. Data Transformation (Joining Weather and Flight dataset)
    - Reduce the size of weather dataset by keeping only the rows relevant to 
      airports in flight dataset.
    - Use external dataset to get weather station ID (IATA code).
    - Join flight data with weather data.
2. Feature engineering.
3. Fill missing values and split the data into tran/test set for ML pipeline, 
   and save them into blob storage.
   
Originally written as a DataBricks notebook so some functions may raise name
errors in plain python (e.g. "display").
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from pyspark.sql.functions import isnull, when, count
from pyspark.sql.functions import *
from pyspark.sql import SQLContext
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.window import Window

from pyspark.sql import Window
from pyspark.sql.functions import rank, col, monotonically_increasing_id
import pyspark
import time
from pyspark.ml.feature import Imputer


blob_container = "w261team8rocks" # Name of container created in https://portal.azure.com
storage_account = "dataguru" # Name of Storage account created in https://portal.azure.com
secret_scope = "XXXXXXX" # Name of the scope created in local computer using the Databricks CLI
secret_key = "XXXXXXX" # Name of the secret key created in local computer using the Databricks CLI
blob_url = f"wasbs://w261team8rocks@dataguru.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# SAS Token
spark.conf.set(
  f"fs.azure.sas.w261team8rocks.dataguru.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)


def data_transformation():
    """Transform the provided raw dataset and save the ouput to blob storage."""
    
    # Load station table
    df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
    # Zero distance to neighbor means the station is the same as its neighbor.
    df_stations = df_stations.filter(col("distance_to_neighbor") == 0)
    print(f"Stations original n = {df_stations.count()}")
    df_stations.createOrReplaceTempView("stations")
    
    # Load external data to map weather station to IATA codes:
    adf = pd.read_csv('https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat', header=None)
    col_names = {
        0: 'AirportID', 1: 'Name', 2: 'City', 3: 'Country', 4: 'IATA', 5: 'ICAO', 6: 'Latitude', 
        7: 'Longitude', 8: 'Altitude', 9: 'Timezone', 10: 'DST', 11: 'TZ_Timezone', 12: 'Type', 13: 'Source'
    }
    adf.rename(columns=col_names, inplace=True)
    df_airport = spark.createDataFrame(adf)
    df_airport.createOrReplaceTempView("airports")
    
    # Join station data with external IATA data
    query_station_airport = """
    SELECT * 
    FROM 
    (SELECT * FROM stations) AS s 
    LEFT JOIN 
    (SELECT ICAO, IATA, Country, Timezone, DST, TZ_Timezone, Altitude FROM airports) AS a
    ON s.neighbor_call = a.ICAO
    """
    stations_with_iata = spark.sql(query_station_airport)
    print(f"Stations joined n = {stations_with_iata.count()}")
    
    # Write final stations dataset to parquet:
    stations_with_iata.write.mode("overwrite").parquet(f"{blob_url}/stations_with_iata")
    
    # Load flight data and get unique airports
    df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
    origin_airports = df_airlines.select("ORIGIN").distinct().collect()
    dest_airports = df_airlines.select("DEST").distinct().collect()
    all_airports = set([o["ORIGIN"] for o in origin_airports] + [d["DEST"] for d in dest_airports])
    unique_airports = spark.createDataFrame([[a] for a in sorted(all_airports)], ["AIRPORT"])
    unique_airports.write.mode("overwrite").parquet(f"{blob_url}/unique_airports")
    
    # Filter stations to only the airports from the full flights dataset:
    airports = {r["AIRPORT"] for r in unique_airports.select("AIRPORT").distinct().collect()}
    print(f"Airports in flights dataset n = {len(airports)}")
    stations_with_iata = stations_with_iata.filter(stations_with_iata.IATA.isin(airports))
    print(f"Airports found in joined stations n = {stations_with_iata.count()}")
    airports_in_joined = {r["IATA"] for r in stations_with_iata.select("IATA").distinct().collect()}
    airports_not_found = airports - airports_in_joined
    print(f"Airports not found: {', '.join(sorted(airports_not_found))}")
    display(stations_with_iata)
    
    # Look at the external data for the missing airports:
    missing_icao = set(adf.loc[(adf["IATA"].isin(airports_not_found)), "ICAO"])
    print('missing airports', missing_icao)
    display(adf.loc[(adf["IATA"].isin(airports_not_found))])
    
    # Look at the counts of flights from airports not found:
    flights_from_bad_airports = df_airlines.filter(df_airlines["ORIGIN"].isin(airports_not_found))
    print('Check counts of flights from airports not found')
    display(flights_from_bad_airports.groupby("ORIGIN").count())
    
    # Get the unique relevant station IDs from the final station table:
    station_ids = {r["station_id"] for r in stations_with_iata.select(col("station_id")).distinct().collect()}
    
    # Load weather table:
    df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")
    
    original_weather_count = df_weather.count()
    print(f"Original weather n = {original_weather_count}")
    
    # Filter the weather table to only the relevant stations:
    df_weather_filtered = df_weather.filter(df_weather.STATION.isin(station_ids))
    filtered_weather_count = df_weather_filtered.count()
    print(f"Filtered weather n = {filtered_weather_count}")
    print(f"Weather data size reduced by {(1-(filtered_weather_count/original_weather_count))*100:.0f}%")
    
    # Round weather data to nearest hour to merge with flights, and then shift by 2 hours.
    # First shift by -1 minutes (so that rows exactly on the hour aren't shifted 3 hours),
    # then shift by 3 hours so that each row is at least 2 hours from its original timestamp.
    weather_original_columns = df_weather_filtered.columns
    
    # Shift by 2 hours:
    df_weather_filtered = df_weather_filtered.withColumn(
        "shifted_timestamp", df_weather_filtered["DATE"] + expr("INTERVAL -1 MINUTES")
    )
    df_weather_filtered = df_weather_filtered.withColumn(
        "shifted_timestamp", df_weather_filtered["shifted_timestamp"] + expr("INTERVAL 3 HOURS")
    )
    
    # Truncate hour (i.e. set minutes and everything after to 0):
    df_weather_filtered = df_weather_filtered.withColumn(
        "final_timestamp", date_trunc("hour", df_weather_filtered.shifted_timestamp)
    )
    
    # Rearrange columns:
    df_weather_filtered = df_weather_filtered.select(
        weather_original_columns[:2] + ["final_timestamp"] + weather_original_columns[2:]
    )
    
    # There will be lots of duplicates by station ID and final_timestamp. 
    # Drop duplicates ordered by station ID and original datestamp, to keep the observation closest to the 
    # final_timestamp. This method for dropping duplicates adapted from:
    # https://stackoverflow.com/a/54738843/6286540
    window = Window.partitionBy("STATION", "final_timestamp").orderBy("DATE", "tiebreak")
    df_weather_deduped = df_weather_filtered\
        .withColumn("tiebreak", monotonically_increasing_id())\
        .withColumn("rank", rank().over(window))\
        .filter(col("rank") == 1).drop("rank", "tiebreak")
    
    print('Weather data after dropping duplicates')
    display(df_weather_deduped)
    
    # Merge relevant weather data with station info:
    
    weather_keep_columns = [
        "STATION", "DATE", "final_timestamp", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", 
        "WND", "CIG", "VIS", "TMP", "DEW", "SLP", 
    ]
    
    station_keep_cols = [
        "station_id", "neighbor_state", "ICAO", "IATA", "Country", "Timezone", "DST", "TZ_Timezone", "Altitude"
    ]
    
    stations_with_iata.select(station_keep_cols).createOrReplaceTempView("stations")
    df_weather_deduped.select(weather_keep_columns).createOrReplaceTempView("weather")
    
    query_weather_stations = f"""
    SELECT * 
    FROM 
    (SELECT {', '.join(weather_keep_columns)} FROM weather) AS w
    LEFT JOIN 
    (SELECT {', '.join(station_keep_cols)} FROM stations) AS s
    ON w.STATION = s.station_id
    """
    
    joined_weather_stations = spark.sql(query_weather_stations)
    print('Joined weather and station data')
    display(joined_weather_stations)  
    
    # Drop irrelevant flight table columns:
    # (e.g. dropped "Gate Return Information at Origin Airport (Data starts 10/2008)" and 
    # "Diverted Airport Information (Data starts 10/2008)" sections)
    flights_keep_columns = [
        'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE',
        'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM',
        'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME',
        'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 
        'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME', 
        'DEST_STATE_ABR', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC',
        'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 
        'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 
        'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK',
        'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 
        'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 
        'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
    ]
    flights = df_airlines.select(*flights_keep_columns)
    
    # Additional clean up to drop flight records:
    
    # Assumption 1: Remove cancelled flights
    # When DEP_DEL15.isNull(), these are cancelled flights which are approximately 1.5% of total flights. Our Phase I 
    # results indicated that flights are cancelled due to the following reasons ("A": "Carrier", "B": "Weather", 
    # "C": "National Air System", "D": "Security"). We can safely take out cancelled flights (null value) from the 
    # delayed flights ("DEP_DEL15") since they are not relevant and minimal.
    flights = flights.where(col("CANCELLED") != 1)
    
    # Assumption 2: Remove diverted flights
    # A flight diversion is when an aircraft is unable to arrive at its final destination. Such as Aircraft emergency; 
    # Passenger emergency; Mechanical failure; and Poor weather conditions. We decided to remove this since it's not 
    # relevant to our analysis.
    flights = flights.where(col("DIVERTED") != 1)
    
    # Drop any duplicate rows in full dataset:
    flights = flights.dropDuplicates()
    
    # There are an additional 4725 rows where for some reason the departure delay columns are null.
    # On inspection, in all these rows the scheduled CRS_DEP_TIME is equal to the DEP_TIME, meaning delay is 0 minutes.
    # Hence we fill these columns with 0:
    flights = flights.fillna(value=0, subset=["DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15", "DEP_DELAY_GROUP"])
    
    # Add origin and destination timezone columns to flights data:
    stations_with_iata.select(["IATA", "Timezone", "TZ_Timezone"]).createOrReplaceTempView("timezones")
    
    # Origin:
    flights.createOrReplaceTempView("flights")
    query_flights_timezone = f"""
    SELECT * 
    FROM 
    (SELECT * FROM flights) AS f
    LEFT JOIN 
    (SELECT IATA AS ORIGIN_IATA, Timezone AS ORIGIN_Timezone, TZ_Timezone AS ORIGIN_TZ FROM timezones) AS tz
    ON f.ORIGIN = tz.ORIGIN_IATA
    """
    flights = spark.sql(query_flights_timezone)
    
    # Destination:
    flights.createOrReplaceTempView("flights")
    query_flights_timezone = f"""
    SELECT * 
    FROM 
    (SELECT * FROM flights) AS f
    LEFT JOIN 
    (SELECT IATA AS DEST_IATA, Timezone AS DEST_Timezone, TZ_Timezone AS DEST_TZ FROM timezones) AS tz
    ON f.DEST = tz.DEST_IATA
    """
    flights = spark.sql(query_flights_timezone)
    
    # Convert flight Departure Times to UTC and round to nearest hour:
    
    # Convert departure time integers to zero-padded strings, e.g. 607 -> 0000607:
    # Modification: Use scheduled departure time instead
    flights = flights.withColumn("PADDED_DEP_TIME", format_string("0000%d", "CRS_DEP_TIME"))
    # Shorten the strings to the final 4 chars, e.g. 0000607 -> 0607:
    flights = flights.withColumn("FORMATTED_DEP_TIME", substring("PADDED_DEP_TIME", -4,4))
    # Concatenate string columns for departure date and time:
    flights = flights.withColumn("DEPT_DT_STR", concat_ws(" ", flights.FL_DATE, flights.FORMATTED_DEP_TIME))
    # Convert string datetime to timestamp:
    flights = flights.withColumn("DEPT_DT", to_timestamp(flights.DEPT_DT_STR, "yyyy-MM-dd HHmm"))
    # Use datetime and timezone to convert dates to UTC:
    flights = flights.withColumn("DEPT_UTC", to_utc_timestamp(flights.DEPT_DT, flights.ORIGIN_TZ))
    # Remove minutes and round datetimes *down* to nearest hour. It is necessary to round
    # down so that we don't join with weather data from less than 2 hours before:
    flights = flights.withColumn("DEPT_UTC_HOUR", date_trunc("HOUR", flights.DEPT_UTC))
    
    # Calculate arrival time in UTC using departure time and elapsed time:
    # Modification: Use scheduled elapsed time
    flights = flights.withColumn("ARR_UTC", col("DEPT_UTC") + (col("CRS_ELAPSED_TIME") * expr("Interval 1 Minutes")))  
    
    # Join flights and weather data (origin and destination) on airport and time:
    flights.createOrReplaceTempView("flights")
    origin_weather_stations = joined_weather_stations.select(
        'final_timestamp', 'IATA', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP'
    )
    origin_weather_stations = origin_weather_stations\
        .withColumnRenamed('IATA', 'weather_ORIGIN_IATA')\
        .withColumnRenamed('final_timestamp', 'DEPT_UTC_HOUR_ORIGIN')\
        .withColumnRenamed('SLP', 'SLP_ORIGIN')\
        .withColumnRenamed('WND', 'WND_ORIGIN')\
        .withColumnRenamed('CIG', 'CIG_ORIGIN')\
        .withColumnRenamed('VIS', 'VIS_ORIGIN')\
        .withColumnRenamed('TMP', 'TMP_ORIGIN')\
        .withColumnRenamed('DEW', 'DEW_ORIGIN')
    destination_weather_stations = joined_weather_stations.select(
        'final_timestamp', 'IATA', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP'
    )
    destination_weather_stations = destination_weather_stations\
        .withColumnRenamed('IATA', 'weather_DEST_IATA')\
        .withColumnRenamed('final_timestamp', 'DEPT_UTC_HOUR_DEST')\
        .withColumnRenamed('SLP', 'SLP_DEST')\
        .withColumnRenamed('WND', 'WND_DEST')\
        .withColumnRenamed('CIG', 'CIG_DEST')\
        .withColumnRenamed('VIS', 'VIS_DEST')\
        .withColumnRenamed('TMP', 'TMP_DEST')\
        .withColumnRenamed('DEW', 'DEW_DEST')
    
    origin_weather_stations.createOrReplaceTempView("ORIGIN_weather")
    destination_weather_stations.createOrReplaceTempView("DEST_weather")  
    
    final_df = spark.sql("""
    select *
    from flights as f
    inner join ORIGIN_weather as ow on f.ORIGIN = ow.weather_ORIGIN_IATA AND f.DEPT_UTC_HOUR = ow.DEPT_UTC_HOUR_ORIGIN
    inner join DEST_weather as dw on f.DEST = dw.weather_DEST_IATA AND f.DEPT_UTC_HOUR = dw.DEPT_UTC_HOUR_DEST
    """)
    
    # Get statistics of the final dataset
    final_n = final_df.count()
    flights_n = flights.count()
    print(f"Final dataset has {final_n:,} rows ({flights_n-final_n:,} dropped from original flights dataset)")
    
    # Split weather columns into individual data columns and deal with null-coded values:
    
    # Wind:
    final_df = final_df\
        .withColumn("WND_dir_ORIGIN", split(final_df["WND_ORIGIN"], ",").getItem(0)) \
        .withColumn("WND_dir_qlty_ORIGIN", split(final_df["WND_ORIGIN"], ",").getItem(1)) \
        .withColumn("WND_type_ORIGIN", split(final_df["WND_ORIGIN"], ",").getItem(2)) \
        .withColumn("WND_spd_ORIGIN", split(final_df["WND_ORIGIN"], ",").getItem(3)) \
        .withColumn("WND_spd_qlty_ORIGIN", split(final_df["WND_ORIGIN"], ",").getItem(4))\
        .withColumn("WND_dir_DEST", split(final_df["WND_DEST"], ",").getItem(0)) \
        .withColumn("WND_dir_qlty_DEST", split(final_df["WND_DEST"], ",").getItem(1)) \
        .withColumn("WND_type_DEST", split(final_df["WND_DEST"], ",").getItem(2)) \
        .withColumn("WND_spd_DEST", split(final_df["WND_DEST"], ",").getItem(3)) \
        .withColumn("WND_spd_qlty_DEST", split(final_df["WND_DEST"], ",").getItem(4))
    
    # CIG:
    final_df = final_df\
        .withColumn("CIG_ceil_height_ORIGIN", split(final_df["CIG_ORIGIN"], ",").getItem(0)) \
        .withColumn("CIG_ceil_qlty_ORIGIN", split(final_df["CIG_ORIGIN"], ",").getItem(1)) \
        .withColumn("CIG_ceil_det_code_ORIGIN", split(final_df["CIG_ORIGIN"], ",").getItem(2)) \
        .withColumn("CIG_cavok_code_ORIGIN", split(final_df["CIG_ORIGIN"], ",").getItem(3))\
        .withColumn("CIG_ceil_height_DEST", split(final_df["CIG_DEST"], ",").getItem(0)) \
        .withColumn("CIG_ceil_qlty_DEST", split(final_df["CIG_DEST"], ",").getItem(1)) \
        .withColumn("CIG_ceil_det_code_DEST", split(final_df["CIG_DEST"], ",").getItem(2)) \
        .withColumn("CIG_cavok_code_DEST", split(final_df["CIG_DEST"], ",").getItem(3))
    
    # Visibility:
    final_df = final_df\
        .withColumn("VIS_dim_ORIGIN", split(final_df["VIS_ORIGIN"], ",").getItem(0)) \
        .withColumn("VIS_dim_qlty_ORIGIN", split(final_df["VIS_ORIGIN"], ",").getItem(1)) \
        .withColumn("VIS_var_ORIGIN", split(final_df["VIS_ORIGIN"], ",").getItem(2)) \
        .withColumn("VIS_var_qlty_ORIGIN", split(final_df["VIS_ORIGIN"], ",").getItem(3))\
        .withColumn("VIS_dim_DEST", split(final_df["VIS_DEST"], ",").getItem(0)) \
        .withColumn("VIS_dim_qlty_DEST", split(final_df["VIS_DEST"], ",").getItem(1)) \
        .withColumn("VIS_var_DEST", split(final_df["VIS_DEST"], ",").getItem(2)) \
        .withColumn("VIS_var_qlty_DEST", split(final_df["VIS_DEST"], ",").getItem(3))
    
    # Temperature:
    final_df = final_df\
        .withColumn("TMP_air_ORIGIN", split(final_df["TMP_ORIGIN"], ",").getItem(0)) \
        .withColumn("TMP_air_qlty_ORIGIN", split(final_df["TMP_ORIGIN"], ",").getItem(1))\
        .withColumn("TMP_air_DEST", split(final_df["TMP_DEST"], ",").getItem(0)) \
        .withColumn("TMP_air_qlty_DEST", split(final_df["TMP_DEST"], ",").getItem(1))
    
    # Dew:
    final_df = final_df\
        .withColumn("DEW_point_temp_ORIGIN", split(final_df["DEW_ORIGIN"], ",").getItem(0)) \
        .withColumn("DEW_point_qlty_ORIGIN", split(final_df["DEW_ORIGIN"], ",").getItem(1))\
        .withColumn("DEW_point_temp_DEST", split(final_df["DEW_DEST"], ",").getItem(0)) \
        .withColumn("DEW_point_qlty_DEST", split(final_df["DEW_DEST"], ",").getItem(1))
    
    # Sea-level pressure:
    final_df = final_df\
        .withColumn("SLP_pressure_ORIGIN", split(final_df["SLP_ORIGIN"], ",").getItem(0)) \
        .withColumn("SLP_pressure_qlty_ORIGIN", split(final_df["SLP_ORIGIN"], ",").getItem(1))\
        .withColumn("SLP_pressure_DEST", split(final_df["SLP_DEST"], ",").getItem(0)) \
        .withColumn("SLP_pressure_qlty_DEST", split(final_df["SLP_DEST"], ",").getItem(1))  
    
    # Replace null-codes:
    col_null_codes = {
        "WND_dir_ORIGIN": "999",
        "WND_type_ORIGIN": "9",
        "WND_spd_ORIGIN": "9999",
        "CIG_ceil_height_ORIGIN": "99999",
        "CIG_ceil_det_code_ORIGIN": "9",
        "CIG_cavok_code_ORIGIN": "9",
        "VIS_dim_ORIGIN": "999999",
        "VIS_var_ORIGIN": "9",
        "TMP_air_ORIGIN": "+9999", 
        "DEW_point_temp_ORIGIN": "+9999",
        "SLP_pressure_ORIGIN": "99999",
        
        "WND_dir_DEST": "999",
        "WND_type_DEST": "9",
        "WND_spd_DEST": "9999",
        "CIG_ceil_height_DEST": "99999",
        "CIG_ceil_det_code_DEST": "9",
        "CIG_cavok_code_DEST": "9",
        "VIS_dim_DEST": "999999",
        "VIS_var_DEST": "9",
        "TMP_air_DEST": "+9999", 
        "DEW_point_temp_DEST": "+9999",
        "SLP_pressure_DEST": "99999"
    }
    for col_name, null_code in col_null_codes.items():
        final_df = final_df.replace(null_code, value=None, subset=[col_name])
    
    # Convert columns types:
    float_cols = [
        "WND_dir_ORIGIN", "WND_spd_ORIGIN", "CIG_ceil_height_ORIGIN", "VIS_dim_ORIGIN", "TMP_air_ORIGIN", 
        "DEW_point_temp_ORIGIN", "SLP_pressure_ORIGIN",
        "WND_dir_DEST", "WND_spd_DEST", "CIG_ceil_height_DEST", "VIS_dim_DEST", "TMP_air_DEST", "DEW_point_temp_DEST",
        "SLP_pressure_DEST"
    ]
    for f_col in float_cols:
        final_df = final_df.withColumn(f_col, final_df[f_col].cast(FloatType()))  
    
    # Save full final dataset:
    final_df.write.mode("overwrite").parquet(f"{blob_url}/team8_full_dataset_V2")
    
    return 'Weather, Station and Flight Dataset are successfully joined'  
  

def get_avg_delay(flight_data):
    """flight_data is assumed to have departure time in UTC and truncated down 
    to nearest hour. Output is a spark dataframe with schema: 
        ORIGIN, 6_hour_before_departure, 2_hour_before_departure, avg_delay
    
    Join the original flight data with output spark data frame by:
        ORIGIN, 6_hour_before_departure, 2_hour_before_departure
    """
    transformed_flight_data = flight_data\
        .withColumn('6_hour_before_departure', flight_data['DEPT_UTC_HOUR'] - expr('INTERVAL 6 hours'))\
        .withColumn('2_hour_before_departure', flight_data['DEPT_UTC_HOUR'] - expr('INTERVAL 2 hours'))
    transformed_flight_data.createOrReplaceTempView('flight_temp')
    
    delay_df = spark.sql(
    """
    select f2.ORIGIN, f2.6_hour_before_departure, f2.2_hour_before_departure, avg(f1.DEP_DELAY) as avg_delay
    from flight_temp as f1
    inner join flight_temp as f2 on (f1.DEPT_UTC_HOUR between f2.6_hour_before_departure and 
    f2.2_hour_before_departure) and (f1.ORIGIN = f2.ORIGIN)
    group by 1,2,3
    order by 1,2,3
    """
    )
    return delay_df


def feature_engineering():
    """Add new features to the transformed dataset. The output includes full set 
    of features that will be used to build the models.
    """
    # Feature 1: average delay
    # This feature needs to be calculated year by year otherwise it takes very long to run. Splitting by years and 
    # concatenating later can improve processing time significantly.
    
    full_dataset = spark.read.parquet(f"{blob_url}/team8_full_dataset_V2")
    for y in [2015, 2016, 2017, 2018, 2019]:
        start_time = time.time()
        print(f'Year {y}')
        # Include one day from previous year to make sure average delay is calculated correctly for January 1st.
        year_data = full_dataset.filter(
            (full_dataset['DEPT_UTC_HOUR'] >= f'{y - 1}-12-31') & 
            (full_dataset['DEPT_UTC_HOUR'] < f'{y + 1}-1-1')
        ) 
        delay_df = get_avg_delay(year_data)
        delay_df = delay_df.filter(delay_df['6_hour_before_departure'] >= f'{y}-1-1') # remove data that doesn't belong to current year
        delay_df.write.mode("overwrite").parquet(f"{blob_url}/avg_delay_{y}_v2")
        print('Done')
        print("--- %s seconds ---" % (time.time() - start_time))
    
    
    # Generate keys to join with transformed dataset
    full_dataset = full_dataset\
        .withColumn('6_hour_before_departure', full_dataset['DEPT_UTC_HOUR'] - expr('INTERVAL 6 hours'))\
        .withColumn('2_hour_before_departure', full_dataset['DEPT_UTC_HOUR'] - expr('INTERVAL 2 hours'))
    
    # Decompose the transformed dataset by year and join the average delay by year to significantly reduce runtime.
    data_2015 = full_dataset.filter(year(full_dataset['DEPT_UTC_HOUR']) == 2015)
    data_2016 = full_dataset.filter(year(full_dataset['DEPT_UTC_HOUR']) == 2016)
    data_2017 = full_dataset.filter(year(full_dataset['DEPT_UTC_HOUR']) == 2017)
    data_2018 = full_dataset.filter(year(full_dataset['DEPT_UTC_HOUR']) == 2018)
    data_2019 = full_dataset.filter(year(full_dataset['DEPT_UTC_HOUR']) == 2019)
    
    # Rename the columns of delay dataframes. Later they will be used as key to join other dataframes and get average 
    # delay for both origin and destination
    delay_2015_ORIGIN = spark.read.parquet(f"{blob_url}/avg_delay_2015_v2")
    delay_2015_ORIGIN = delay_2015_ORIGIN.withColumnRenamed("avg_delay","avg_delay_ORIGIN")
    delay_2016_ORIGIN = spark.read.parquet(f"{blob_url}/avg_delay_2016_v2")
    delay_2016_ORIGIN = delay_2016_ORIGIN.withColumnRenamed("avg_delay","avg_delay_ORIGIN")
    delay_2017_ORIGIN = spark.read.parquet(f"{blob_url}/avg_delay_2017_v2")
    delay_2017_ORIGIN = delay_2017_ORIGIN.withColumnRenamed("avg_delay","avg_delay_ORIGIN")
    delay_2018_ORIGIN = spark.read.parquet(f"{blob_url}/avg_delay_2018_v2")
    delay_2018_ORIGIN = delay_2018_ORIGIN.withColumnRenamed("avg_delay","avg_delay_ORIGIN")
    delay_2019_ORIGIN = spark.read.parquet(f"{blob_url}/avg_delay_2019_v2")
    delay_2019_ORIGIN = delay_2019_ORIGIN.withColumnRenamed("avg_delay","avg_delay_ORIGIN")
    
    delay_2015_DEST = spark.read.parquet(f"{blob_url}/avg_delay_2015_v2")
    delay_2015_DEST = delay_2015_DEST\
        .withColumnRenamed("avg_delay","avg_delay_DEST")\
        .withColumnRenamed("ORIGIN","DEST")
    delay_2016_DEST = spark.read.parquet(f"{blob_url}/avg_delay_2016_v2")
    delay_2016_DEST = delay_2016_DEST\
        .withColumnRenamed("avg_delay","avg_delay_DEST")\
        .withColumnRenamed("ORIGIN","DEST")
    delay_2017_DEST = spark.read.parquet(f"{blob_url}/avg_delay_2017_v2")
    delay_2017_DEST = delay_2017_DEST\
        .withColumnRenamed("avg_delay","avg_delay_DEST")\
        .withColumnRenamed("ORIGIN","DEST")
    delay_2018_DEST = spark.read.parquet(f"{blob_url}/avg_delay_2018_v2")
    delay_2018_DEST = delay_2018_DEST \
        .withColumnRenamed("avg_delay","avg_delay_DEST") \
        .withColumnRenamed("ORIGIN","DEST")
    delay_2019_DEST = spark.read.parquet(f"{blob_url}/avg_delay_2019_v2")
    delay_2019_DEST = delay_2019_DEST\
        .withColumnRenamed("avg_delay","avg_delay_DEST")\
        .withColumnRenamed("ORIGIN","DEST")    
    
    # Join the split dataset with average delay data (by year) on both origin and destination
    data_2015 = data_2015.join(
        delay_2015_ORIGIN, 
        on = ['ORIGIN','2_hour_before_departure','6_hour_before_departure'],
        how = 'left')\
        .join(
        delay_2015_DEST, 
        on = ['DEST','2_hour_before_departure','6_hour_before_departure'],
        how = 'left'
    )
    
    data_2016 = data_2016.join(
        delay_2016_ORIGIN, 
        on = ['ORIGIN','2_hour_before_departure','6_hour_before_departure'],
        how = 'left')\
        .join(
        delay_2016_DEST, 
        on = ['DEST','2_hour_before_departure','6_hour_before_departure'],
        how = 'left'
    )
    
    data_2017 = data_2017.join(
        delay_2017_ORIGIN, 
        on = ['ORIGIN','2_hour_before_departure','6_hour_before_departure'],
        how = 'left')\
        .join(
        delay_2017_DEST, 
        on = ['DEST','2_hour_before_departure','6_hour_before_departure'],
        how = 'left'
    )
    
    data_2018 = data_2018.join(
        delay_2018_ORIGIN, 
        on = ['ORIGIN','2_hour_before_departure','6_hour_before_departure'],
        how = 'left')\
        .join(
        delay_2018_DEST, 
        on = ['DEST','2_hour_before_departure','6_hour_before_departure'],
        how = 'left'
    )
    
    data_2019 = data_2019.join(
        delay_2019_ORIGIN, 
        on = ['ORIGIN','2_hour_before_departure','6_hour_before_departure'],
        how = 'left')\
        .join(
        delay_2019_DEST, 
        on = ['DEST','2_hour_before_departure','6_hour_before_departure'],
        how = 'left'
    )    
    
    data = data_2015.union(data_2016).union(data_2017).union(data_2018).union(data_2019)
    
    # Feature 2: Prior Flight Delay & Potential for Delay Indicator
    # (1) Previous Flight Delay Indicator (Categorical value) - Is delayed = 1; not delayed = 0; otherwise null.
    # (2) Potential for Delay Indicator (Categorical value) - Set to null if flight arrives more than 2 hrs before 
    # departure, the likelihood for delay is smaller; Set to 1 if flights arrives less than 2 hrs before departure.
    
    # First filter for rows where actual arrival date is greater than departure date 
    data = data.withColumn(
        "ARR_UTC", f.when(
            (data.ARR_UTC < data.DEPT_UTC),
            (f.from_unixtime(f.unix_timestamp('DEPT_UTC') + (data.ACTUAL_ELAPSED_TIME*60)))
        ).otherwise(data.ARR_UTC))
    
    # Group by tail number, then sort by actual arrival time
    tail_group = Window.partitionBy('tail_num').orderBy('ARR_UTC')

    # Flag for 1 if previous flight is delayed for the same airplane (identified by tail number)
    data = data\
        .withColumn('prev_actual_arr_utc', f.lag('ARR_UTC',1, None).over(tail_group))\
        .withColumn('prev_fl_del', f.lag('DEP_DEL15',1, None).over(tail_group))

    # Categorize flight gap (>2 hours = 0, < 2 hours = 1) Has the airplane arrived 2 hours before departure? Simplify 
    # to 1 if airplane is in the airport less than 2 hours before departure, otherwise 0 if not or null.
    data = data\
        .withColumn("planned_departure_utc", col("DEPT_UTC") - (col("DEP_DELAY") * expr("Interval 1 Minutes")))\
        .withColumn('inbtwn_fl_hrs', (f.unix_timestamp('planned_departure_utc') - f.unix_timestamp('prev_actual_arr_utc'))/60/60)\
        .withColumn('poten_for_del', expr("CASE WHEN inbtwn_fl_hrs > 2 THEN '0'" + "ELSE '1' END")) 
    
    #Feature 3: Holiday Indicator
    data = data\
        .withColumn('holiday', expr(
        """CASE WHEN FL_DATE in (
            '2015-12-25', '2016-12-25', '2017-12-25', '2018-12-25', '2019-12-25',
            '2015-11-26', '2016-11-24', '2017-11-23', '2018-11-22', '2019-11-28', 
             '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01',
             '2015-07-04', '2016-07-04', '2017-07-04', '2018-07-04', '2019-07-04') 
        THEN 'holiday' """ + 
        """WHEN FL_DATE in (
            '2015-12-23', '2015-12-24', '2015-12-26', '2015-12-27',
            '2016-12-23', '2016-12-24', '2016-12-26', '2016-12-27',
            '2017-12-23', '2017-12-24', '2017-12-26', '2017-12-27',
            '2018-12-23', '2018-12-24', '2018-12-26', '2018-12-27',
            '2019-12-23', '2019-12-24', '2019-12-26', '2019-12-27',
            '2015-11-24', '2015-11-25', '2015-11-27', '2015-11-28',
            '2016-11-22', '2016-11-24', '2016-11-25', '2016-11-26',
            '2017-11-21', '2017-11-22', '2017-11-24', '2017-11-25',
            '2018-11-20', '2018-11-21', '2018-11-23', '2018-11-24',
            '2019-11-26', '2019-11-27', '2019-11-29', '2019-11-30', 
            '2015-01-02', '2015-01-03', '2015-12-30', '2015-12-31',
            '2016-01-02', '2016-01-03', '2016-12-30', '2016-12-31',
            '2017-01-02', '2017-01-03', '2017-12-30', '2017-12-31',
            '2018-01-02', '2018-01-03', '2018-12-30', '2018-12-31',
            '2019-01-02', '2019-01-03', '2019-12-30', '2019-12-31',
            '2015-07-02', '2015-07-03', '2015-07-05', '2015-07-06',
            '2016-07-02', '2016-07-03', '2016-07-05', '2016-07-06',
            '2017-07-02', '2017-07-03', '2017-07-05', '2017-07-06',
            '2018-07-02', '2018-07-03', '2018-07-05', '2018-07-06',
            '2019-07-02', '2019-07-03', '2019-07-05', '2019-07-06')
         THEN 'nearby_holiday' """ + 
        "ELSE 'non-holiday' END"))
    
    # Feature 4: departure hour in local time
    data = data.withColumn('local_departure_hour', hour(data.DEPT_DT))
    # This dataset includes all original columns from the joined dataset as well as the engineered features. No extra 
    # encoding or transformation is included. This dataset may contain missing values.
    data.write.mode('overwrite').parquet(f"{blob_url}/full_dataset_full_features_v2")  
    return 'Full feature dataset is now generated'


def fill_missing(train, test):
    """Fill missing values in training set for:
        1. Numeric values: using median of the column (this operation will be 
           achieved by pyspark.ml.feature.Imputer)
        2. Categorical values: adding "NA" value to make a new categorical value
    
    Then the exactly same transformation will be applied to the test set.
    """
    categorical_cols = [
        'QUARTER',
        'MONTH',
        'DAY_OF_MONTH',
        'DAY_OF_WEEK',
        'FL_DATE',
        'OP_CARRIER',
        
        'ORIGIN_STATE_ABR',
        'DEST_STATE_ABR',
        
        'WND_type_ORIGIN', 'WND_type_DEST',
        'VIS_var_ORIGIN', 'VIS_var_DEST',          
        
        'poten_for_del', 
        
        'holiday','local_departure_hour'
    ]
    
    numeric_cols = [
        'WND_dir_ORIGIN', 'WND_dir_DEST',
        'WND_spd_ORIGIN', 'WND_spd_DEST',
        'VIS_dim_ORIGIN', 'VIS_dim_DEST',
        'TMP_air_ORIGIN', 'TMP_air_DEST',    
        'DEW_point_temp_ORIGIN', 'DEW_point_temp_DEST',
        'SLP_pressure_ORIGIN', 'SLP_pressure_DEST',    
        'CRS_ELAPSED_TIME',
        'DISTANCE',
        'avg_delay_ORIGIN', 'avg_delay_DEST'   
    ]
    # Impute missing numeric value by median of the column
    imputer = Imputer(
        strategy='median',inputCols = numeric_cols,
        outputCols=["{}_imputed".format(c) for c in numeric_cols]
    )
    imputer_model = imputer.fit(train)
    
    transformed_train = imputer_model.transform(train)
    transformed_test = imputer_model.transform(test)
    
    # Replace the old columns
    for c in numeric_cols:
        transformed_train = transformed_train.drop(c)
        transformed_train = transformed_train.withColumnRenamed(c+'_imputed', c)
        transformed_test = transformed_test.drop(c)
        transformed_test = transformed_test.withColumnRenamed(c+'_imputed', c)
    
    # Assign arbitrary value to missing categorical values
    transformed_train = transformed_train.fillna(value='NA',subset = categorical_cols)
    transformed_test = transformed_test.fillna(value='NA',subset = categorical_cols) 
    # prev_fl_del is binary, fillna with 0
    transformed_train = transformed_train.fillna(value=0,subset = ['prev_fl_del'])
    transformed_test = transformed_test.fillna(value=0,subset = ['prev_fl_del']) 
    return transformed_train,transformed_test


def generate_ml_dataset():
    """First drop columns that are irrelevant to models, then fill missing values, 
    and finally split it into train and test set.
    """ 
    t8_full = spark.read.parquet(f"{blob_url}/full_dataset_full_features_v2")
    
    # Keep relevant columns
    t8_reduced = t8_full.select(
        'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 
        
        'OP_CARRIER',
    
        'ORIGIN_STATE_ABR',
        'DEST_STATE_ABR',
    
        'DEP_DEL15',
        'CRS_ELAPSED_TIME',
        'DISTANCE',
    
        'WND_dir_ORIGIN', 'WND_dir_DEST',
        'WND_spd_ORIGIN', 'WND_spd_DEST',
        'VIS_dim_ORIGIN', 'VIS_dim_DEST',
        'TMP_air_ORIGIN', 'TMP_air_DEST',
        'DEW_point_temp_ORIGIN', 'DEW_point_temp_DEST',
        'SLP_pressure_ORIGIN', 'SLP_pressure_DEST',
        'WND_type_ORIGIN', 'WND_type_DEST',
        'VIS_var_ORIGIN', 'VIS_var_DEST',  
    
        'avg_delay_ORIGIN', 'avg_delay_DEST',  
        
        'prev_fl_del', 'poten_for_del',
        
        'holiday', 'local_departure_hour'
    )
    
    # Drop rows where target variable is missing
    t8_reduced = t8_reduced.dropna(subset="DEP_DEL15")
    
    # Make split into train/test set
    t8_ML_train = t8_reduced.filter(t8_reduced['YEAR'] < 2019)
    t8_ML_test = t8_reduced.filter(t8_reduced['YEAR'] == 2019)
    
    # Fill missing values
    t8_ML_train_filled, t8_ML_test_filled = fill_missing(t8_ML_train, t8_ML_test)  
    
    # Save processed data
    t8_ML_train_filled.write.mode("overwrite").parquet(f"{blob_url}/ML_train_filled")
    t8_ML_test_filled.write.mode("overwrite").parquet(f"{blob_url}/ML_test_filled")
    return 'train/test data have been created'


if __name__ == "__main__":
    
    # Preprocess the data and join tables
    data_transformation()
    # Apply feature engineering
    feature_engineering()
    # Split the data and apply the transformation (fill missing values)
    generate_ml_dataset()
