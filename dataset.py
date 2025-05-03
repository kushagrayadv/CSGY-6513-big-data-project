import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset

def read_sensor_file(filepath):
    df = pd.read_csv(filepath, header=None, names=["timestamp", "activity", "x", "y", "z"])
    df.dropna(inplace=True)
    df["timestamp"] = df["timestamp"].astype(np.int64)
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

def create_labeled_windows(features, label, window_size, step_size):
    X, y = [], []
    for i in range(0, len(features) - window_size + 1, step_size):
        X.append(features[i:i+window_size])
        y.append(label)
    return X, y

class WISDMUnifiedDataset(Dataset):
    def __init__(self, base_dir, freq_hz=50, window_size=100, step_size=50):
        self.X = []
        self.y = []
        self.device = []

        for device_name in ['phone', 'watch']:
            for sensor_type in ['accelerometer', 'gyroscope']:
                if not os.path.exists(os.path.join(base_dir, device_name, sensor_type)):
                    continue

            filenames = sorted(f for f in os.listdir(os.path.join(base_dir, device_name, 'accelerometer')) if f.endswith(".txt"))

            for fname in filenames:
                sid = fname.split("_")[0]
                accel_path = os.path.join(base_dir, device_name, "accelerometer", f"{sid}_accel.txt")
                gyro_path  = os.path.join(base_dir, device_name, "gyroscope", f"{sid}_gyro.txt")

                if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
                    continue

                accel_df = read_sensor_file(accel_path)
                gyro_df  = read_sensor_file(gyro_path)

                common_activities = set(accel_df["activity"].unique()) & set(gyro_df["activity"].unique())

                for activity in common_activities:
                    acc_block = accel_df[accel_df["activity"] == activity]
                    gyr_block = gyro_df[gyro_df["activity"] == activity]

                    start = max(acc_block['timestamp'].min(), gyr_block['timestamp'].min())
                    end   = min(acc_block['timestamp'].max(), gyr_block['timestamp'].max())

                    if end - start < 1e9:
                        continue

                    timestamps = create_common_timestamp_grid(start, end, freq_hz)

                    acc_i = interpolate_sensor_block(acc_block, timestamps)
                    gyr_i = interpolate_sensor_block(gyr_block, timestamps)

                    features = np.stack([
                        acc_i['x'], acc_i['y'], acc_i['z'],
                        gyr_i['x'], gyr_i['y'], gyr_i['z']
                    ], axis=1)

                    X_seqs, Y_seqs = create_labeled_windows(features, activity, window_size, step_size)
                    self.X.extend(X_seqs)
                    self.y.extend(Y_seqs)
                    self.device.extend([0 if device_name == 'phone' else 1] * len(Y_seqs))

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)
        self.device = torch.tensor(np.array(self.device), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.device[idx]


dataset = WISDMUnifiedDataset(base_dir="path/to/WISDM")

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for x, label, device in loader:
    print(x.shape)      # (32, 100, 6)
    print(label.shape)  # (32,)
    print(device.shape) # (32,)
    break
