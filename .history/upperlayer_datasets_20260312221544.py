import os
import pandas as pd
import numpy as np
from tqdm import tqdm 

INPUT_ROOT = os.path.join('..', 'TrafficData', 'datasets_csv_add2', 'datasets_split')
OUTPUT_ROOT = os.path.join('..', 'TrafficData', 'datasets_csv_add2', 'datasets_upperlayer')

CHUNKSIZE = 200000


def filter_upperlayer_packets(input_csv, output_csv):
    """
    Filter packets that contain at least one upper-layer field
    (tcp.options.* or tls.*). Results are written incrementally.
    """

    first_chunk = True
    total = 0
    kept = 0

    for chunk in tqdm(pd.read_csv(
        input_csv,
        dtype=str,
        chunksize=CHUNKSIZE,
        low_memory=False
    )):

        total += len(chunk)

        upper_cols = [
            c for c in chunk.columns
            if c.startswith("tcp.options.") or c.startswith("tls.")
        ]

        if len(upper_cols) == 0:
            raise RuntimeError(
                f"No tcp.options.* or tls.* columns found in {input_csv}"
            )

        # 检查是否至少存在一个 upper-layer 字段
        mask = chunk[upper_cols].apply(
            lambda row: np.any((row.notna()) & (row != "0") & (row != "")),
            axis=1
        )

        filtered = chunk[mask]
        kept += len(filtered)

        if first_chunk:
            filtered.to_csv(output_csv, index=False)
            first_chunk = False
        else:
            filtered.to_csv(output_csv, mode="a", header=False, index=False)

    print(f"{os.path.basename(input_csv)}: {total} -> {kept}")


def process_all_datasets():

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for dataset_name in os.listdir(INPUT_ROOT):

        dataset_path = os.path.join(INPUT_ROOT, dataset_name)

        if not os.path.isdir(dataset_path):
            continue

        print(f"\nProcessing dataset: {dataset_name}")

        output_dataset_path = os.path.join(OUTPUT_ROOT, dataset_name)
        os.makedirs(output_dataset_path, exist_ok=True)

        for split in ["train_set.csv", "val_set.csv", "test_set.csv"]:

            input_csv = os.path.join(dataset_path, split)

            if not os.path.exists(input_csv):
                print(f"Skip missing file: {input_csv}")
                continue

            output_csv = os.path.join(output_dataset_path, split)

            filter_upperlayer_packets(input_csv, output_csv)


if __name__ == "__main__":
    process_all_datasets()