import glob
import os

import numpy as np
import torch
from pyspark.sql import SparkSession, functions as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm


class WISDMUnifiedDataset(IterableDataset):
    def __init__(self, parquet_paths, window_size=100, num_features=6):
        self.parquet_paths = parquet_paths
        self.window_size = window_size
        self.num_features = num_features

        # Create one SparkSession per process
        self.spark = SparkSession.builder \
            .appName("WISDM-SparkLoader") \
            .getOrCreate()

        self.df = (self.spark
                   .read
                   .parquet(*parquet_paths)
                   )

        label_set = set(self.df.select('label').distinct().toPandas()['label'])
        self.class_names = sorted(label_set)  # e.g. ['A', …, 'S']
        self.name2idx = {n: i for i, n in enumerate(self.class_names)}
        self.idx2name = {i: n for n, i in self.name2idx.items()}
        self.num_classes = len(self.class_names)

    def __len__(self):
        return self.df.count()

    def __iter__(self):
        print("iter calling")
        # 1) Read + shuffle in Spark
        df = self.df.orderBy(F.rand())  # global shuffle each epoch
        # 2) Stream rows partition-by-partition
        for row in df.toLocalIterator():
            flat = np.array(row.features, dtype=np.float32)
            seq = flat.reshape(self.window_size, self.num_features)
            yield torch.from_numpy(seq), torch.tensor(self.name2idx[row.label]), torch.tensor(row.device)


if __name__ == "__main__":
    # Testing the dataset
    data_path = os.getcwd() + '/data/wisdm-dataset/processed'
    train_paths = glob.glob(os.path.join(data_path, "train/train_*.parquet"))
    val_paths = glob.glob(os.path.join(data_path, "val/val_*.parquet"))

    train_dataset = WISDMUnifiedDataset(
        parquet_paths=train_paths,
        window_size=100,
        num_features=6
    )

    val_dataset = WISDMUnifiedDataset(
        parquet_paths=val_paths,
        window_size=100,
        num_features=6
    )

    print(f"Length of training dataset: {len(train_dataset)}, val dataset: {len(val_dataset)}")

    BATCH_SIZE = 32
    num_epochs = 1
    num_train_batches = len(train_dataset) // BATCH_SIZE
    num_val_batches = len(val_dataset) // BATCH_SIZE

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Spark sessions aren’t fork‐safe
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True
    )

    for epoch in range(num_epochs):
        for x, y, d in tqdm(train_loader,
                            total=num_train_batches,
                            leave=False):
            # x: (batch, 100, 6), y: (batch,), d: (batch,)
            pass
            # … your training step …

        for x, y, d in tqdm(val_loader,
                            total=num_val_batches,
                            leave=False):
            # x: (batch, 100, 6), y: (batch,), d: (batch,)
            pass
            # … your validation step …

        # next epoch → __iter__ called again → fresh Spark shuffle
