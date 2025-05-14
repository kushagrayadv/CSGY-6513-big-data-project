import math
import os
import random
import json
import asyncio
import numpy as np
from pyspark.sql import SparkSession, functions as F, types as T, Row
import requests
import zipfile
import io
import tempfile
import glob
from aiokafka import AIOKafkaProducer
import scipy
from scipy.interpolate import interp1d

import utils

FREQ_HZ = 50
WINDOW_SIZE = 100
STEP_SIZE = 50


def get_spark_session():
    return SparkSession.builder.appName("WISDM_Stream_Preproc").config("spark.driver.memory", "10g").getOrCreate()


def process_subjects(subject_set, spark, device_name, device_id, base_dir='./data/raw/wisdm-dataset/raw'):
    """
    Returns list of (features_list, label_str, device_id, subject_id) rows
    """
    rows = []
    accel_dir = os.path.join(base_dir, device_name, "accel")
    gyro_dir = os.path.join(base_dir, device_name, "gyro")

    if not os.path.exists(accel_dir) or not os.path.exists(gyro_dir):
        print(f"Error: Directory not found - {accel_dir} or {gyro_dir}")
        return rows

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
            print(f"[WARNING] Missing files for subject {subj_id}")
            continue

        try:
            accel_df = utils.read_sensor_file(spark, accel_path, schema=df_schema)
            gyro_df = utils.read_sensor_file(spark, gyro_path, schema=df_schema)

            activities = (set(row.activity for row in accel_df.select("activity").distinct().collect())
                        & set(row.activity for row in gyro_df.select("activity").distinct().collect()))

            for activity in activities:
                print(f"Processing activity {activity} for subj {subj_id}...")
                accel_pdf = accel_df.filter(F.col("activity") == activity).toPandas().sort_values(
                    "timestamp").drop_duplicates(subset="timestamp")
                gyro_pdf = gyro_df.filter(F.col("activity") == activity).toPandas().sort_values(
                    "timestamp").drop_duplicates(subset="timestamp")

                if len(accel_pdf) < 2 or len(gyro_pdf) < 2:
                    print(f"[WARNING] Not enough samples for activity {activity}")
                    continue

                start = max(accel_pdf.timestamp.min(), gyro_pdf.timestamp.min())
                end = min(accel_pdf.timestamp.max(), gyro_pdf.timestamp.max())
                if end - start < 1_000_000_000:  # <1s overlap
                    print(f"[WARNING] Insufficient overlap for activity {activity}")
                    continue

                timestamps = utils.create_common_timestamp_grid(start, end, freq_hz=FREQ_HZ)
                accel_i = utils.interpolate_sensor_block(df=accel_pdf, timestamps=timestamps)
                gyro_i = utils.interpolate_sensor_block(df=gyro_pdf, timestamps=timestamps)

                features = np.stack([accel_i["x"], accel_i["y"], accel_i["z"], 
                                  gyro_i["x"], gyro_i["y"], gyro_i["z"]], axis=1)
                
                windows = utils.create_labeled_windows(
                    features=features,
                    label=activity,
                    device_id=device_id,
                    subject_id=subj_id,
                    window_size=WINDOW_SIZE,
                    step_size=STEP_SIZE
                )
                rows.extend(windows)
                print(f"Added {len(windows)} windows for activity {activity}")

        except Exception as e:
            print(f"Error processing subject {subj_id}: {str(e)}")
            continue

    print(f"\nTotal windows generated for {device_name}: {len(rows)}")
    return rows


def rows_to_df(spark, rows):
    # schema: features: array<double>, label: string, device: int, subject_id: string
    return spark.createDataFrame(
        [Row(features=r[0].flatten().tolist(), label=r[1], device=r[2], subject_id=r[3]) for r in rows]
    )


def get_train_test__split(base_dir='./data/raw/wisdm-dataset/raw'):
    phone_acc = os.path.join(base_dir, "phone/accel")
    phone_gyr = os.path.join(base_dir, "phone/gyro")
    watch_acc = os.path.join(base_dir, "watch/accel")
    watch_gyr = os.path.join(base_dir, "watch/gyro")

    # Download and extract dataset if not exists
    if not os.path.exists(base_dir):
        os.makedirs("./data/raw", exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"
        print("Downloading dataset...")
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download dataset: HTTP {response.status_code}")
        
        zip_path = "./data/raw/wisdm-dataset.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./data/raw")
        print("Dataset downloaded and extracted")

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


class DataProducer:
    def __init__(self, bootstrap_servers='localhost:9092', topic_name='sensor_data'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic_name
        self.producer = None
        self.rows_sent = 0
        self.spark = None

    async def start(self):
        """Start the Kafka producer"""
        try:
            # Initialize Spark session
            self.spark = get_spark_session()

            # Initialize Kafka producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            await self.producer.start()

            # Read processed data
            phone_df = self.spark.read.parquet("./data/processed/phone_test.parquet")
            watch_df = self.spark.read.parquet("./data/processed/watch_test.parquet")

            # Print schema and sample data
            print("\nPhone DataFrame Schema:")
            phone_df.printSchema()
            print("\nPhone DataFrame Sample:")
            phone_df.show(5)

            print("\nWatch DataFrame Schema:")
            watch_df.printSchema()
            print("\nWatch DataFrame Sample:")
            watch_df.show(5)

            # Convert to pandas for easier iteration
            phone_data = phone_df.toPandas()
            watch_data = watch_df.toPandas()

            print("\nPhone Pandas Columns:", phone_data.columns.tolist())
            print("Watch Pandas Columns:", watch_data.columns.tolist())

            print(f"Loaded {len(phone_data)} phone records and {len(watch_data)} watch records")
            return True

        except Exception as e:
            print(f"Error in producer: {e}")
            if self.producer:
                await self.producer.stop()
            if self.spark:
                self.spark.stop()
            return False

    async def send_data(self):
        """Send data to Kafka topic"""
        try:
            phone_df = self.spark.read.parquet("./data/processed/phone_test.parquet")
            watch_df = self.spark.read.parquet("./data/processed/watch_test.parquet")

            # Convert to pandas for easier iteration
            phone_data = phone_df.toPandas()
            watch_data = watch_df.toPandas()

            print(f"\nLoaded {len(phone_data)} phone records and {len(watch_data)} watch records")

            # Process phone data by subject and activity
            for device_type, data in [('phone', phone_data), ('watch', watch_data)]:
                print(f"\nProcessing {device_type} data:")
                
                # Group data by subject and activity
                for subject_id in sorted(set(data['subject_id'])):
                    subject_data = data[data['subject_id'] == subject_id]
                    print(f"\nSubject {subject_id}:")
                    
                    for activity in sorted(set(subject_data['label'])):
                        activity_data = subject_data[subject_data['label'] == activity]
                        activity_count = len(activity_data)
                        print(f"  Activity {activity}: {activity_count} windows")
                        
                        # Send windows for this subject and activity
                        for _, row in activity_data.iterrows():
                            features = np.array(row['features']).reshape(WINDOW_SIZE, 6)
                            
                            # Skip windows containing NaN values
                            if np.isnan(features).any():
                                continue
                                
                            message = {
                                'device': device_type,
                                'subject_id': str(subject_id),
                                'activity': str(activity),
                                'features': features.tolist()
                            }
                            await self.producer.send_and_wait(self.topic, message)
                            self.rows_sent += 1

            print(f"\nTotal messages sent: {self.rows_sent}")

        except Exception as e:
            print(f"Error sending data: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def run(self):
        """Main producer loop"""
        try:
            if not await self.start():
                return

            await self.send_data()

        except Exception as e:
            print(f"Error in producer: {e}")
            raise
        finally:
            if self.producer:
                await self.producer.stop()
            if self.spark:
                self.spark.stop()


def ensure_dataset_available():
    """Download and extract the dataset if not already available"""
    base_dir = './data/raw/wisdm-dataset/raw'
    if os.path.exists(base_dir):
        print("Dataset already exists")
        return base_dir

    print("Dataset not found. Downloading...")
    os.makedirs("./data/raw", exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        zip_path = "./data/raw/wisdm-dataset.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Dataset downloaded successfully")
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./data/raw")
        print("Dataset extracted successfully")
        
        # Verify the extraction
        if not os.path.exists(base_dir):
            raise Exception(f"Expected directory {base_dir} not found after extraction")
            
        # List contents to verify
        print("\nVerifying dataset structure:")
        for device in ['phone', 'watch']:
            for sensor in ['accel', 'gyro']:
                path = os.path.join(base_dir, device, sensor)
                if os.path.exists(path):
                    files = os.listdir(path)
                    print(f"{device}/{sensor}: {len(files)} files")
                else:
                    print(f"WARNING: Directory not found: {path}")
        
        return base_dir
        
    except Exception as e:
        print(f"Error downloading/extracting dataset: {e}")
        if os.path.exists("./data/raw/wisdm-dataset.zip"):
            os.remove("./data/raw/wisdm-dataset.zip")
        raise


async def main():
    try:
        # Ensure dataset is available
        base_dir = ensure_dataset_available()
        print(f"Using dataset in: {base_dir}")
        
        # Get train/test split
        train_subset, val_subset, test_subset = get_train_test__split(base_dir=base_dir)
        
        # Initialize Spark
        spark = get_spark_session()
        
        print("\nProcessing test subjects for streaming...")
        
        # Process phone data
        phone_rows = process_subjects(test_subset, spark, "phone", 0, base_dir=base_dir)
        if phone_rows:
            print(f"Generated {len(phone_rows)} windows from phone data")
        else:
            print("No phone data was generated")
        
        # Process watch data
        watch_rows = process_subjects(test_subset, spark, "watch", 1, base_dir=base_dir)
        if watch_rows:
            print(f"Generated {len(watch_rows)} windows from watch data")
        else:
            print("No watch data was generated")
        
        if not phone_rows and not watch_rows:
            raise Exception("No data was processed. Check the dataset paths and subjects.")
        
        # Create output directories
        os.makedirs("./data/processed", exist_ok=True)
        
        # Convert to DataFrame and save
        if phone_rows:
            phone_df = rows_to_df(spark, phone_rows)
            output_path = "./data/processed/phone_test.parquet"
            phone_df.write.mode("overwrite").parquet(output_path)
            print(f"Saved phone data to {output_path}")
            
        if watch_rows:
            watch_df = rows_to_df(spark, watch_rows)
            output_path = "./data/processed/watch_test.parquet"
            watch_df.write.mode("overwrite").parquet(output_path)
            print(f"Saved watch data to {output_path}")
        
        print("Data processing complete. Starting Kafka producer...")
        
        # Start Kafka producer
        producer = DataProducer()
        await producer.run()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if 'spark' in locals():
            spark.stop()


if __name__ == "__main__":
    asyncio.run(main())
