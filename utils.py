import numpy as np
from pyspark.sql import functions as F
from scipy.interpolate import interp1d

def read_sensor_file(filepath, device_name, schema):
  df = (
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
    .withColumn("device", F.lit((0 if device_name == 'phone' else 1)))
  )
  return df


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


def create_labeled_windows(features, label, device_name, window_size, step_size):
  X, y, device = [], [], []
  for i in range(0, len(features) - window_size + 1, step_size):
    X.append(features[i:i + window_size])
    y.append(label)
    device.append(device_name)
  return X, y, device