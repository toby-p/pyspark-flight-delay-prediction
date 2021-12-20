# Databricks notebook source
# MAGIC %md
# MAGIC # Phase I - Describe datasets, joins, task, and metrics 
# MAGIC ## Team 8

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, max
import pyspark


def spark_shape(self):
  """Hack to emulate pandas' df.shape"""
  return (self.count(), len(self.columns))

  
pyspark.sql.dataframe.DataFrame.shape = spark_shape

# COMMAND ----------

# Inspect the Mount's Final Project folder 
display(dbutils.fs.ls("/mnt/mids-w261/datasets_final_project"))

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Flight Data Schema 
# MAGIC https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ&flf_gnoyr_anzr=g_bagVZR_eRcbegVaT&h5r4_gnoyr_anzr=er2146v0t%20Pn44vr4%20b0-gvzr%20cr4s14zn0pr%20(EMLK-24r5r06)&lrn4_V0s1=E&Sv456_lrn4=EMLK&Yn56_lrn4=FDFE&en6r_V0s1=D&S4r37r0pB=Z106uyB&Qn6n_S4r37r0pB=N007ny,d7n46r4yB,Z106uyB

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Useful Columns
# MAGIC 1. Time dependent columns: YEAR QUARTER MONTH DAY_OF_MONTH DAY_OF_WEEK FL_DATE
# MAGIC 2. Delay related columns: DepDel15 DepDelay DepDelayMinutes ArrDelay ArrDelayMinutes ArrDel15 Cancelled Diverted CarrierDelay WeatherDelay NASDelay SecurityDelay LateAircraftDelay
# MAGIC 3. Flight related columns: ActualElapsedTime AirTime Distance
# MAGIC 4. Geographical columns: OriginAirportID

# COMMAND ----------

df_airlines.select('DEP_DEL15').distinct().collect()

# COMMAND ----------

# About 22% of flights departed with delays. We will have to deal with class imbalance in the modeling process.
# About 3% of flights don't have labels on delays (They are cancelled flights).
total_flight = df_airlines.count()
delayed_flight = df_airlines.filter(df_airlines.DEP_DEL15 == 1).cache()
on_time_flight = df_airlines.filter(df_airlines.DEP_DEL15 == 0).cache()
cancelled_flight = df_airlines.filter(df_airlines.DEP_DEL15.isNull()).cache()
print(f'Total Flight: {total_flight}')
print(f'Delayed Flight: {delayed_flight.count() / total_flight }')
print(f'On Time Flight: {on_time_flight.count() / total_flight}')
print(f'Cancelled Flight: {cancelled_flight.count() / total_flight}')
display(df_airlines)

# COMMAND ----------

# MAGIC %md
# MAGIC The unknown flights are all cancellations. Therefore we need to decide if we include these as delayed flights when creating the model. We can try including them and dropping them and see which model performs better: 

# COMMAND ----------

# Convert to pandas as can easily store in memory:
df = df_airlines.toPandas()
df["DELAY_ISNA"] = df["DEP_DELAY_GROUP"].isna()
cancelled = pd.pivot_table(df, index=("DELAY_ISNA"), columns=("CANCELLED"), values="YEAR", aggfunc="count")
cancelled.fillna(0).astype(int)

# COMMAND ----------

# Can also look at the different cancellation reasons:
cancel_raw = df[["CANCELLATION_CODE", "DELAY_ISNA", "YEAR"]].copy().fillna("NaN")
cancellation_reasons = pd.pivot_table(cancel_raw, index=("DELAY_ISNA"), columns=("CANCELLATION_CODE"), 
                                      values="YEAR", aggfunc="count", margins=True, margins_name="TOTAL")
rename = {"A": "Carrier", "B": "Weather", "C": "National Air System", "D": "Security"}
cancellation_reasons.rename(columns=rename).fillna(0).astype(int)

# COMMAND ----------

# The DEP_DELAY_GROUP field groups delay times by 15 minutes intervals.
# A look at the distribution - buckets 2 and above are the outcome, i.e. flights delayed 15m or more:
n_buckets = len(df_airlines.select("DEP_DELAY_GROUP").distinct().collect())
hist = df_airlines.select("DEP_DELAY_GROUP").rdd.flatMap(lambda x: x).histogram(n_buckets)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x=hist[0][1:], height=hist[1])

# COMMAND ----------

# Different Origins have different delay times
print('Departure Delay Average')
display(delayed_flight.groupBy('ORIGIN').avg('DEP_DELAY').collect())
print('Arrival Delay Average')
display(delayed_flight.groupBy('ORIGIN').avg('ARR_DELAY').collect())
display(delayed_flight.groupBy('DAY_OF_WEEK').avg('DEP_DELAY').collect())

# COMMAND ----------

# There seems to be some seasonality in the flight delays
display(delayed_flight)

# COMMAND ----------

# Monday has the most delayed flights
display(delayed_flight)

# COMMAND ----------

# Flights are almost uniformly distributed across the week
display(df_airlines)

# COMMAND ----------

df_airlines.select('ORIGIN').distinct().collect()

# COMMAND ----------

df_airlines.select('DEST_AIRPORT_ID').distinct().count()

# COMMAND ----------

df_airlines.count()

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

df_weather = df_weather.select('NAME').rdd.flatMap(lambda x: (x.strip().split(','))).collect()

# COMMAND ----------

df_weather.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC # Weather data
# MAGIC 1. Schema is available https://docs.google.com/spreadsheets/d/1v0P34NlQKrvXGCACKDeqgxpDwmj3HxaleUiTrY7VRn0/edit#gid=0

# COMMAND ----------

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

result_pdf = df_stations.select("*").toPandas()

# COMMAND ----------

result_pdf.neighbor_name.unique()

# COMMAND ----------

atl= result_pdf[result_pdf["neighbor_name"].str.contains("ATLANTA")]
atl.neighbor_name.unique()


# COMMAND ----------

chi= result_pdf[result_pdf["neighbor_name"].str.contains("CHICAGO")]
chi.neighbor_name.unique() 

# COMMAND ----------

result_pdf[(result_pdf.neighbor_name == 'HARTSFIELD-JACKSON ATLANTA IN') & (result_pdf.distance_to_neighbor == 0)]

# COMMAND ----------

result_pdf[(result_pdf.neighbor_name == "CHICAGO O'HARE INTERNATIONAL") & (result_pdf.distance_to_neighbor == 0)]

# COMMAND ----------



# COMMAND ----------

result_pdf[result_pdf.station_id == '72219013874'].sort_values(by=['distance_to_neighbor']).head(30)

# COMMAND ----------

result_pdf[result_pdf.station_id == '72530094846'].sort_values(by=['distance_to_neighbor']).head(30)

# COMMAND ----------



# COMMAND ----------

# Weather data can be joined with station data by station_id

# COMMAND ----------

# MAGIC %md
# MAGIC # Import CSV Data (Airport ID) from Azure Blob Storage

# COMMAND ----------

from pyspark.sql.functions import col, max

blob_container = "w261team8rocks" # The name of your container created in https://portal.azure.com
storage_account = "dataguru" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261-team8" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "cloudblob" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://w261team8rocks@dataguru.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# SAS Token
spark.conf.set(
  f"fs.azure.sas.w261team8rocks.dataguru.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# Load the Airport ID data for join tables later
df_airport_ID = spark.read.csv(f"{blob_url}/L_AIRPORT_ID.csv")
display(df_airport_ID)

# COMMAND ----------


