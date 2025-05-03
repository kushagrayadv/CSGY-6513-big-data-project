import math
import os
import random

import numpy as np
from data_pipeline import gyro_df
from pyspark.sql import SparkSession, functions as F, types as T, Row

import utils

FREQ_HZ = 50
WINDOW_SIZE = 100
STEP_SIZE = 50


def get_spark_session():
    return SparkSession.builder.appName("WISDM_Split_Preproc").config("spark.driver.memory", "10g").getOrCreate()


def process_subjects(subject_set, spark, device_name, device_id, base_dir='./data'):
    """
    Returns list of (features_list, label_str, device_id) rows
    """
    rows = []
    accel_dir = os.path.join(base_dir, device_name, "accel")
    gyro_dir = os.path.join(base_dir, device_name, "gyro")

    df_schema = T.StructType([
        T.StructField("subject_id", T.StringType(), True),
        T.StructField("activity", T.StringType(), True),
        T.StructField("timestamp", T.LongType(), True),
        T.StructField("x", T.DoubleType(), True),
        T.StructField("y", T.DoubleType(), True),
        T.StructField("z_raw", T.StringType(), True),
    ])

    for subj_id in subject_set:
        print(f"\nProcessing subject {subj_id}...")
        accel_path = os.path.join(accel_dir, f"data_{subj_id}_accel_{device_name}.txt")
        gyro_path = os.path.join(gyro_dir, f"data_{subj_id}_gyro_{device_name}.txt")
        if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
            print("[ERROR] path doesn't exist")
            continue

        accel_df = utils.read_sensor_file(spark, accel_path, schema=df_schema)
        gyro_df = utils.read_sensor_file(spark, gyro_path, schema=df_schema)

        activities = (set(row.activity for row in accel_df.select("activity").distinct().collect())
                      & set(row.activity for row in gyro_df.select("activity").distinct().collect()))

        for activity in activities:
            print(f"Processing activity {activity} for subj {subj_id}...")
            accel_pdf = accel_df.filter(F.col("activity") == activity).toPandas().sort_values(
                "timestamp").drop_duplicates(
                subset="timestamp")
            gyro_pdf = gyro_df.filter(F.col("activity") == activity).toPandas().sort_values(
                "timestamp").drop_duplicates(
                subset="timestamp")

            if len(accel_pdf) < 2 or len(gyro_pdf) < 2:
                continue

            start = max(accel_pdf.timestamp.min(), gyro_pdf.timestamp.min())
            end = min(accel_pdf.timestamp.max(), gyro_pdf.timestamp.max())
            if end - start < 1_000_000_000:  # <1s overlap
                continue

            timestamps = utils.create_common_timestamp_grid(start, end, freq_hz=FREQ_HZ)
            accel_i = utils.interpolate_sensor_block(df=accel_pdf, timestamps=timestamps)
            gyro_i = utils.interpolate_sensor_block(df=gyro_pdf, timestamps=timestamps)

            features = np.stack([accel_i["x"], accel_i["y"], accel_i["z"], gyro_i["x"], gyro_i["y"], gyro_i["z"]],
                                axis=1)
            rows.extend(utils.create_labeled_windows(features=features,
                                                     label=activity,
                                                     device_id=device_id,
                                                     window_size=WINDOW_SIZE, step_size=STEP_SIZE))

    return rows


def rows_to_df(spark, rows):
    # schema: features: array<double>, label: string, device: int
    return spark.createDataFrame(
        [Row(features=r[0], label=r[1], device=r[2]) for r in rows]
    )


def get_train_test__split(base_dir='./data'):
    phone_acc = os.path.join(base_dir, "phone/accel")
    phone_gyr = os.path.join(base_dir, "phone/gyro")
    watch_acc = os.path.join(base_dir, "watch/accel")
    watch_gyr = os.path.join(base_dir, "watch/gyro")

    all_subs = sorted({f.split("_")[1]
                       for f in os.listdir(phone_acc) if f.endswith(".txt")})
    random.seed(42)
    random.shuffle(all_subs)

    n = len(all_subs)
    n_train = math.ceil(0.8 * n)
    n_val = math.floor(0.1 * n)

    train_subset = set(all_subs[:n_train])
    val_subset = set(all_subs[n_train:n_train + n_val])
    test_subset = set(all_subs[n_train + n_val:])

    return train_subset, val_subset, test_subset


if __name__ == "__main__":
    base_dir = os.getcwd() + '/data/wisdm-dataset/raw'
    train_subs, val_subs, test_subs = get_train_test__split(base_dir=base_dir)
    print(f"Train subjects: {len(train_subs)}, Val: {len(val_subs)}, Test: {len(test_subs)}")

    train_rows = []
    val_rows = []

    spark = get_spark_session()

    for device_name, device_id in [("phone", 0), ("watch", 1)]:
        print(f"\n==============Processing subjects for device {device_name}=============")
        tr = process_subjects(train_subs, spark, device_name, device_id)
        vr = process_subjects(val_subs, spark, device_name, device_id)
        train_rows.extend(tr)
        val_rows.extend(vr)

    spark.stop()
    print(f"Total training samples: {len(train_rows)} | Total validation samples: {len(val_rows)}")

    processed_data_dir = os.getcwd() + '/data/wisdm-dataset/processed'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    os.makedirs(os.path.join(processed_data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, "val"), exist_ok=True)

    # Start a new spark session for saving .parquet files
    spark = get_spark_session()

    slice_size = 10_000
    part = 0
    for i in range(0, len(train_rows), slice_size):
        print(f"Saving train part {part:03d}, rows {i}-{i + slice_size - 1}")
        rows_to_df(spark, train_rows[i: i + slice_size]).write.mode("overwrite").parquet(
            os.makedirs(os.path.join(processed_data_dir, f"train/train_data_{part:03d}.parquet")))
        part += 1

    rows_to_df(spark, val_rows).write.mode("overwrite").parquet(
        os.path.join(processed_data_dir, "val/val_data.parquet"))

    print("Train & Val windows written to parquet.")
    print("Test subjects held out for Kafka streaming:", sorted(test_subs))

    spark.stop()
