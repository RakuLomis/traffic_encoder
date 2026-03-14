import os
import pandas as pd
import numpy as np
from tqdm import tqdm 

INPUT_ROOT = os.path.join('..', 'TrafficData', 'datasets_csv_add2', 'datasets_split')
OUTPUT_ROOT = os.path.join('..', 'TrafficData', 'datasets_csv_add2', 'datasets_upperlayer')

CHUNKSIZE = 200000


SUMMARY_FILE = "upperlayer_filter_summary.csv"


def count_lines(csv_path):
    """Count total rows for progress bar"""
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def filter_upperlayer_packets(input_csv, output_csv):

    total_lines = count_lines(input_csv)

    first_chunk = True
    total = 0
    kept = 0

    reader = pd.read_csv(
        input_csv,
        dtype=str,
        chunksize=CHUNKSIZE,
        low_memory=False
    )

    with tqdm(total=total_lines, desc=os.path.basename(input_csv)) as pbar:

        for chunk in reader:

            chunk_size = len(chunk)
            total += chunk_size

            upper_cols = [
                c for c in chunk.columns
                if c.startswith("tcp.options.") or c.startswith("tls.")
            ]

            if len(upper_cols) == 0:
                raise RuntimeError(
                    f"No tcp.options.* or tls.* columns found in {input_csv}"
                )

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

            pbar.update(chunk_size)

    removed = total - kept
    removal_ratio = removed / total if total > 0 else 0

    return total, kept, removed, removal_ratio


def process_all_datasets():

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    summary_rows = []

    datasets = [
        d for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    for dataset_name in tqdm(datasets, desc="Datasets"):

        dataset_path = os.path.join(INPUT_ROOT, dataset_name)
        output_dataset_path = os.path.join(OUTPUT_ROOT, dataset_name)

        os.makedirs(output_dataset_path, exist_ok=True)

        dataset_stats = {
            "dataset": dataset_name
        }

        # for split in ["train_set.csv", "validation_set.csv", "test_set.csv"]:
        for split in ["validation_set.csv"]:

            input_csv = os.path.join(dataset_path, split)

            if not os.path.exists(input_csv):
                continue

            output_csv = os.path.join(output_dataset_path, split)

            total, kept, removed, ratio = filter_upperlayer_packets(
                input_csv,
                output_csv
            )

            split_name = split.replace(".csv", "")

            dataset_stats[f"{split_name}_total"] = total
            dataset_stats[f"{split_name}_kept"] = kept
            dataset_stats[f"{split_name}_removed"] = removed
            dataset_stats[f"{split_name}_removed_ratio"] = ratio

            print(
                f"{dataset_name} | {split_name}: "
                f"{removed}/{total} removed ({ratio:.2%})"
            )

        summary_rows.append(dataset_stats)

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\nSummary saved to:", SUMMARY_FILE)
    print(summary_df)


if __name__ == "__main__":
    process_all_datasets()