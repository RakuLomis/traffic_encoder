import os
import shutil

from utils.pcap_tools_enhance import convert_pcap_to_raw_csv
from utils.dataframe_tools import (
    consolidate_raw_csvs_memory_optimized,
    build_stream_id_from_five_tuple,
    global_stratified_split_memory_optimized,
    truncate_to_block_by_schema,
    augment_main_block_top_k,
)
from utils.pruning_and_merge import merge_field_blocks_tree_similarity

NEED_CHIEF_BLOCK = False
SPLIT_BY_FLOW = True

def run_full_pipeline(raw_data_root: str, output_root: str, force_overwrite: bool = False):
    """Run end-to-end preprocessing for all datasets under raw_data_root."""
    print("=" * 80)
    print("### Start preprocessing pipeline ###")
    print("=" * 80)

    try:
        dataset_names = [
            d for d in os.listdir(raw_data_root)
            if os.path.isdir(os.path.join(raw_data_root, d))
        ]
    except FileNotFoundError:
        print(f"Error: raw data root not found: {raw_data_root}")
        return

    if not dataset_names:
        print(f"Warning: no dataset directory found under: {raw_data_root}")
        return

    print(f"Found {len(dataset_names)} dataset(s): {dataset_names}\n")

    for dataset_name in dataset_names:
        print("\n" + "#" * 80)
        print(f"### Processing dataset: {dataset_name} ###")
        print("#" * 80)

        paths = {
            "pcap_dir": os.path.join(raw_data_root, dataset_name),
            "raw_csv_dir": os.path.join(output_root, 'datasets_csv', dataset_name),
            "consolidated_csv": os.path.join(output_root, 'datasets_consolidate', f"{dataset_name}.csv"),
            "split_dir": os.path.join(output_root, 'datasets_split', dataset_name),
            "truncated_blocks_dir": os.path.join(output_root, 'datasets_fbt', 'truncation', dataset_name),
            "merged_blocks_dir": os.path.join(output_root, 'datasets_fbt', 'merger', dataset_name),
            "augmented_train_set": os.path.join(output_root, 'datasets_final', f"{dataset_name}_chief_block_augmented.csv"),
        }

        try:
            print("\n>>> Step 1/3: PCAP -> Raw CSVs")
            if (not force_overwrite and os.path.exists(paths['raw_csv_dir'])
                    and os.listdir(paths['raw_csv_dir'])):
                print(f" -> Skip: non-empty output dir exists: {paths['raw_csv_dir']}")
            else:
                if force_overwrite and os.path.exists(paths['raw_csv_dir']):
                    print(f" -> [FORCE] removing old dir: {paths['raw_csv_dir']}")
                    shutil.rmtree(paths['raw_csv_dir'])
                convert_pcap_to_raw_csv(paths['pcap_dir'], paths['raw_csv_dir'])
            print(" -> Step 1 done.")

            print("\n>>> Step 2/3: Raw CSVs -> Consolidated CSV")
            if not force_overwrite and os.path.exists(paths['consolidated_csv']):
                print(f" -> Skip: consolidated file exists: {paths['consolidated_csv']}")
            else:
                consolidate_raw_csvs_memory_optimized(paths['raw_csv_dir'], paths['consolidated_csv'])
                print("\n>>> Step 2.5/3: Build stream_id and drop tcp.stream")
                stream_tmp_path = paths['consolidated_csv'] + '.tmp_stream_id.csv'
                build_stream_id_from_five_tuple(
                    input_csv_path=paths['consolidated_csv'],
                    output_csv_path=stream_tmp_path,
                    chunksize=200000,
                    drop_tcp_stream=True,
                )
                os.replace(stream_tmp_path, paths['consolidated_csv'])
                print(" -> Step 2.5 done.")
            print(" -> Step 2 done.")

            # print("\n>>> Step 2.5/3: Build stream_id and drop tcp.stream")
            # stream_tmp_path = paths['consolidated_csv'] + '.tmp_stream_id.csv'
            # build_stream_id_from_five_tuple(
            #     input_csv_path=paths['consolidated_csv'],
            #     output_csv_path=stream_tmp_path,
            #     chunksize=200000,
            #     drop_tcp_stream=True,
            # )
            # os.replace(stream_tmp_path, paths['consolidated_csv'])
            # print(" -> Step 2.5 done.")

            print("\n>>> Step 3/3: Final Processing (Split / FBT / Merge / Augment)")
            if NEED_CHIEF_BLOCK:
                if not force_overwrite and os.path.exists(paths['augmented_train_set']):
                    print(f" -> Skip: final train set exists: {paths['augmented_train_set']}")
                else:
                    print("\n  -> Sub-step 3.1: Stratified split")
                    global_stratified_split_memory_optimized(
                        paths['consolidated_csv'],
                        paths['split_dir'],
                        split_by_flow=SPLIT_BY_FLOW,
                        flow_id_col='stream_id',
                        stratify_by_flow_length=True,
                        flow_length_bins=4,
                    )

                    print("\n  -> Sub-step 3.2: Field block truncation")
                    train_set_path = os.path.join(paths['split_dir'], 'train_set.csv')
                    truncate_to_block_by_schema(train_set_path, paths['truncated_blocks_dir'])

                    print("\n  -> Sub-step 3.3: Merge field blocks")
                    merge_field_blocks_tree_similarity(
                        paths['truncated_blocks_dir'],
                        paths['merged_blocks_dir'],
                        similarity_threshold=0.8,
                    )

                    print("\n  -> Sub-step 3.4: Chief block augmentation")
                    os.makedirs(os.path.dirname(paths['augmented_train_set']), exist_ok=True)
                    augment_main_block_top_k(paths['merged_blocks_dir'], paths['augmented_train_set'])
            else:
                if not force_overwrite and os.path.exists(paths['split_dir']):
                    print(f" -> Skip: split dir exists: {paths['split_dir']}")
                else:
                    print("\n  -> Sub-step 3.1: Stratified split")
                    global_stratified_split_memory_optimized(
                        paths['consolidated_csv'],
                        paths['split_dir'],
                        split_by_flow=True,
                        flow_id_col='stream_id',
                        stratify_by_flow_length=True,
                        flow_length_bins=4,
                    )

            print(" -> Step 3 done.")
            print(f"\n### Dataset '{dataset_name}' processed successfully ###")

        except Exception as e:
            print(f"\n!!!!!! Fatal error while processing '{dataset_name}': {e} !!!!!!")
            print("!!!!!! Continue to next dataset. !!!!!!")
            continue

    print("\n" + "=" * 80)
    print("### All datasets processed ###")
    print("=" * 80)


if __name__ == '__main__':
    input_root = os.path.join('..', 'TrafficData', 'datasets_raw_add2')
    output_root = os.path.join('..', 'TrafficData', 'datasets_csv_add2')
    run_full_pipeline(input_root, output_root)
