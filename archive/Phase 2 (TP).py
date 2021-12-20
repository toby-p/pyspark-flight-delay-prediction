# Databricks notebook source
# MAGIC %md
# MAGIC # Phase II - EDA, Scalability, Efficiency, Distributed/parallel Training and Scoring Pipeline
# MAGIC ## Toby Petty - Team 8
# MAGIC 
# MAGIC ### Problem description:
# MAGIC 
# MAGIC Flight delays create problems in scheduling for airlines and airports, leading to passenger inconvenience, and huge economic losses. As a result there is growing interest in predicting flight delays beforehand in order to optimize operations and improve customer satisfaction. In this project, you will be predicting flight delays using the datasets provided. For now, the problem to be tackled in this project is framed as follows:
# MAGIC Predict departure delay/no delay, where a delay is defined as 15-minute delay (or greater) with respect to the planned time of departure. This prediction should be done two hours ahead of departure (thereby giving airlines and airports time to regroup and passengers a heads up on a delay). 
# MAGIC 
# MAGIC ### Phase II description
# MAGIC 
# MAGIC * EDA on all tables 
# MAGIC * Join tables and generate the dataset that will be used for training and evaluation
# MAGIC * Joins takes 2-3 hours with 10 nodes; 
# MAGIC * Join stations data with flights data
# MAGIC * Join weather data with flights + Stations data
# MAGIC * Store on cold blob storage (on overwrite mode) [Azure: free credit: $100; use it for storage only]  
# MAGIC * EDA on joined dataset that will be used for training and evaluation
# MAGIC * Address missing data
# MAGIC * Address non-numerical features
# MAGIC * List out raw  features, derived features that you plan to implement/use 
# MAGIC * Do you need any dimensionality reduction? (ie L1, forward/backward selection, PCA, etc..)
# MAGIC * Specify the feature transformations for the pipeline and justify these features given the target (ie, hashing trick, tf-idf, stopword removal, lemmatization, tokenization, etc..)
# MAGIC * Other feature engineering efforts, i.e. interaction terms, Brieman’s method, etc…)
# MAGIC * Baseline modeling and evaluation on the smaller datasets (3 month and 6 month)
# MAGIC * Hint: cross-validation in Time Series (very different to regular cross-validation
# MAGIC * STRETCH Goal: Do a baseline experiment on all the data 

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, max
import pyspark


blob_container = "w261team8rocks"
storage_account = "dataguru"
secret_scope = "w261-team8"
secret_key = "cloudblob"
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)


def spark_shape(self):
  """Hack to emulate pandas' df.shape"""
  return (self.count(), len(self.columns))

pyspark.sql.dataframe.DataFrame.shape = spark_shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stations data
# MAGIC The below cells use an external CSV file found on Github to find weather station IATA codes (e.g. Chicago O'Hare = "ORD") and merge them to the stations dataframe where possible. This means that we can then merge the stations data to the weather data on `station_id`, and then merge the weather data with the airlines data on IATA code = `ORIGIN`.

# COMMAND ----------

def save_airport_codes():
  """Get external file linking airport IATA codes to `neighbor_call` 
  column in stations_df. The data also saves an `elevation_ft` column
  which may be another useful feature."""
  url = "https://raw.githubusercontent.com/datasets/airport-codes/master/data/airport-codes.csv"
  df = pd.read_csv(url)
  columns = ["gps_code", "iata_code", "elevation_ft"]
  df = df.dropna(subset=["gps_code", "iata_code"])[columns]
  df = df.sort_values(by=["gps_code"]).reset_index(drop=True)
  df = spark.createDataFrame(df)
  df.write.mode("overwrite").parquet(f"{blob_url}/airport_codes")
  return df

# COMMAND ----------

airport_codes = save_airport_codes()
display(airport_codes)

# COMMAND ----------

def save_stations():
  """Save neighbor_call column for each station_id."""
  df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
  columns = ["station_id", "neighbor_call"]
  stations = df_stations.filter(col("distance_to_neighbor") == 0).select(*station_cols).dropDuplicates()
  stations.createOrReplaceTempView("stations")
  
  # Join the IATA airport codes:
  airport_codes = spark.read.parquet(f"{blob_url}/airport_codes")
  airport_codes.createOrReplaceTempView("airport_codes")
  query = """
  SELECT * FROM {0} s
  LEFT JOIN {1} a
  ON s.neighbor_call == a.gps_code
  """.format("stations", "airport_codes")
  merged = spark.sql(query)
  
  # Save to data store:
  merged.write.mode("overwrite").parquet(f"{blob_url}/stations")
  return merged

# COMMAND ----------

stations = save_stations()

# Sanity check:
display(stations.filter(col("iata_code") == "ORD"))
display(df_stations.filter(col("neighbor_call") == "KORD"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Airlines data

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

def preprocess_airlines_df(df):
  """Apply preprocessing steps to airlines dataframes."""
  
  # Drop columns:
  # ========================================================================
  all_columns = list(df.columns)
  
  # Columns relating to diverted flights - they aren't counted as 
  # delayed, and are diverted after take-off anyway:
  cols_to_drop = [c for c in all_columns if c.startswith("DIV")]
  
  # Year/Month/Day columns that duplicate data in flight date:
  cols_to_drop += ["YEAR", "MONTH", "DAY_OF_MONTH"]
  
  # Airport ID columns that change over time:
  cols_to_drop += [c for c in all_columns if "AIRPORT_SEQ_ID" in c]
  
  cols_to_keep = [c for c in all_columns if c not in cols_to_drop]
  df = df.select(cols_to_keep)
  
  return df


df = preprocess_airlines_df(df_airlines)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")

# Save a small sample for testing joins:
weather_sample = df_weather.filter(col("DATE") <= "2015-01-02T00:00:00.000")

# COMMAND ----------

weather_sample.write.mode("overwrite").parquet(f"{blob_url}/weather_sample")

# COMMAND ----------

def join_weather_and_stations(stations_name: str = "stations", 
                              weather_name: str = "weather_sample"):
  """
  Args:
    stations_name: name of stations data saved in blob_url.
    weather_name: name of weather data saved in blob_url.
  """
  stations = spark.read.parquet(f"{blob_url}/{stations_name}")
  stations.createOrReplaceTempView("stations")
  weather = spark.read.parquet(f"{blob_url}/{weather_name}")
  weather.createOrReplaceTempView("weather")
  
  query = """
  SELECT * FROM {0} w
  LEFT JOIN {1} s
  ON w.STATION == s.station_id
  """.format("weather", "stations")
  
  return spark.sql(query)

# COMMAND ----------

joined = join_weather_and_stations()
display(joined.filter(col("iata_code") == "ORD"))

# COMMAND ----------

# MAGIC %md
# MAGIC # SCRATCH BELOW

# COMMAND ----------

columns = list(df_airlines.columns)

def search_column(s: str):
  """Helper function to look for columns containing a specified substring."""
  columns_lower = {str.lower(c): c for c in columns}
  return [v for k, v in columns_lower.items() if s.lower() in k]

search_column("iata")

# COMMAND ----------

df_airlines.dtypes

# COMMAND ----------

df_airlines.select("DISTANCE_GROUP", "DEP_DELAY").groupby("DISTANCE_GROUP").mean().sort("DISTANCE_GROUP").collect()

# COMMAND ----------

def describe_missing(df: pd.DataFrame):
  counts, n = df.count(), len(df)
  fraction_notna = (counts / n).sort_values(ascending=False)
  full = fraction_notna[fraction_notna == 1]
  print(f"{len(counts)} columns in dataframe")
  print(f"{len(full):,} columns of {len(counts):,} have no missing values ({len(full) / len(counts) * 100:.2f}%)")
  partial = fraction_notna[fraction_notna < 1]
  fig, ax = plt.subplots(figsize=(12, 3))
  ax.bar(x = range(len(partial)), height=partial.values)
  ax.set_xticks(range(len(partial)))
  ax.set_xticklabels(partial.index, rotation=90)
  ax.set_title(f"Fraction complete of {len(partial):,} columns missing some values")

# COMMAND ----------

describe_missing(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Target feature - delay

# COMMAND ----------

def search_column(s: str, df: pd.DataFrame):
  """Helper function to look for columns containing a specified substring."""
  columns_lower = {str.lower(c): c for c in df.columns}
  return [v for k, v in columns_lower.items() if s.lower() in k]

search_column("delay", df)

# COMMAND ----------

search_column("delay", df)

# COMMAND ----------

# MAGIC %md
# MAGIC There is a numeric column `DEP_DELAY` which has the actual delay in minutes, and a categorical column `DEP_DELAY_GROUP` which has the delay bucketed into 15 minute intervals. Check that the data is consistent with this categorization scheme:

# COMMAND ----------

pd.pivot_table(df, index=("DEP_DELAY_GROUP"), values="DEP_DELAY", aggfunc=("mean", "min", "max", "count"))

# COMMAND ----------

# MAGIC %md
# MAGIC What about NaN values? Can they be explained by cancelled flights? Do we include cancellations?

# COMMAND ----------

df["DELAY_ISNA"] = df["DEP_DELAY_GROUP"].isna()
cancelled = pd.pivot_table(df, index=("DELAY_ISNA"), columns=("CANCELLED"), values="YEAR", aggfunc="count")
cancelled.fillna(0).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC Handful of rows where `DEP_DELAY_GROUP` is NaN but `CANCELLED` = 1. What's going on with those rows? Maybe flights that departed the terminal (i.e. taxi'd) but didn't actually take off?

# COMMAND ----------

cancel_raw = df[["CANCELLATION_CODE", "DELAY_ISNA", "YEAR"]].copy().fillna("NaN")
cancellation_reasons = pd.pivot_table(cancel_raw, index=("DELAY_ISNA"), columns=("CANCELLATION_CODE"), 
                                      values="YEAR", aggfunc="count", margins=True, margins_name="TOTAL")
rename = {"A": "Carrier", "B": "Weather", "C": "National Air System", "D": "Security"}
cancellation_reasons.rename(columns=rename).fillna(0).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC For target feature create binary feature which is 1 for `DEP_DELAY_GROUP >=1`, 0 otherwise. 
# MAGIC 
# MAGIC Check the distribution of `DEP_DELAY_GROUP` and the balance of this binary feature:

# COMMAND ----------

df["TARGET"] = df["DEP_DELAY_GROUP"] >= 1
df["TARGET"] = np.where(df["DEP_DELAY_GROUP"].isna(), "NaN", df["TARGET"])
counts = df["TARGET"].value_counts()
counts / len(df)

# COMMAND ----------

# MAGIC %md
# MAGIC There is a class imbalance, so we should use F1-score / weighted F1-score as the performance metric.
# MAGIC 
# MAGIC Need to decide what to do with ~3% of flights which are cancelled.

# COMMAND ----------

n_buckets = len(df_airlines.select("DEP_DELAY_GROUP").distinct().collect())
hist = df_airlines.select("DEP_DELAY_GROUP").rdd.flatMap(lambda x: x).histogram(n_buckets)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x=hist[0][1:], height=hist[1])

# COMMAND ----------

fig, ax = plt.subplots(figsize=(8, 5))
_ = ax.hist(df["DEP_DELAY_GROUP"], bins=len(df["DEP_DELAY_GROUP"].unique()))
ax.set_title(f"Delay Group Distribution")

# COMMAND ----------

# Handful of rows where flight is cancelled, but there is no DELAY value:
df[(df["DEP_DELAY"].notna()) & (df["CANCELLED"] == 1)]

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

spark_shape(df_weather)

# COMMAND ----------

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

query = """
SELECT * FROM {0} weather
LEFT JOIN {1} stations
ON weather.station_id == stations.station_id
""".format("weather_sample", "stations")
query

# COMMAND ----------



# COMMAND ----------

stations.write.mode("overwrite").parquet(f"{blob_url}/stations")

# COMMAND ----------

display(dbutils.fs.ls(f"{blob_url}"))

# COMMAND ----------

weather_sample = df_weather.filter(col("DATE") <= "2015-01-02T00:00:00.000")
spark_shape(weather_sample)

# COMMAND ----------

display(weather_sample)

# COMMAND ----------

display(stations.select("station_id").groupby("station_id").count().sort("count", ascending=False))

# COMMAND ----------

blob_container = "w261team8rocks"
storage_account = "dataguru"
secret_scope = "w261-team8"
secret_key = "cloudblob"
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

display(dbutils.fs.ls(f"{mount_path}"))

# COMMAND ----------

tp_test_df = sc.parallelize([("a", 1), ("b", 2)]).toDF(["col1", "col2"])
tp_test_df.write.parquet(f"{blob_url}/tp_test_small")

# COMMAND ----------

display(dbutils.fs.ls(f"{blob_url}"))

# COMMAND ----------

# This command will write to your Cloud Storage if right permissions are in place. 
# Navigate back to your Storage account in https://portal.azure.com, to inspect the files.
df_weather.write.parquet(f"{blob_url}/weather_data_3m")

# Load it the previous DF as a new DF
df_weather_new = spark.read.parquet(f"{blob_url}/weather_data_3m/*")
display(df_weather_new)

print(f"Your new df_weather has {df_weather_new.count():,} rows.")
print(f'Max date: {df_weather_new.select([max("DATE")]).collect()[0]["max(DATE)"].strftime("%Y-%m-%d %H:%M:%S")}')

# COMMAND ----------

https://www.transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_NVecbeg_VQ
