import numpy as np
from pyspark.sql import functions as F
from scipy.interpolate import interp1d

def read_sensor_file(spark, filepath, schema):
  return (
    spark.read
    .option("header", "false")
    .schema(schema)
    .csv(filepath)
    .na.drop()
    .withColumn("z",
                F.regexp_replace("z_raw", r'[^0-9\.\-]', "")
                .cast("double"))
    .drop("z_raw")
    .withColumn("timestamp", F.col("timestamp").cast("long"))
  )


def create_common_timestamp_grid(start, end, freq_hz):
  duration = (end - start) / 1e9
  num_samples = int(duration * freq_hz)
  return np.linspace(start, end, num_samples)


def interpolate_sensor_block(df, timestamps):
  interp = {}
  for axis in ['x', 'y', 'z']:
    f = interp1d(df['timestamp'], df[axis], kind='linear', fill_value="extrapolate")
    interp[axis] = f(timestamps)
  return interp


def create_labeled_windows(features, label, device_id, subject_id, window_size=100, step_size=50):
    """Create sliding windows of sensor data with labels"""
    windows = []
    n_samples = len(features)
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window = features[start:end]
        windows.append((window, label, device_id, subject_id))
    
    return windows