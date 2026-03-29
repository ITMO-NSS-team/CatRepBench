import pandas as pd
from pathlib import Path
from typing import List
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

IDS_DIR = SCRIPT_DIR / "dataset_ids"
RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "raw"


def get_ids_from_file(path: Path) -> List[int]:
    """Reads a list of dataset IDs from a file."""
    if not path.exists():
        print(f"[WARNING] ID not found: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [int(line.strip()) for line in f if line.strip()]


def sanitize_filename(name: str) -> str:
    """Removes invalid characters from the filename."""
    return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)


def process_openml_datasets() -> int:
    """Downloads and saves datasets from OpenML. Returns the count of
    successful downloads."""
    print("Downloading datasets from OpenML...")
    dataset_ids = get_ids_from_file(IDS_DIR / "openml.txt")
    success_count = 0

    for dataset_id in dataset_ids:
        try:
            print(f"  Processing OpenML ID: {dataset_id}")
            dataset = fetch_openml(data_id=dataset_id, as_frame=True,
                                   parser='auto')
            df = dataset.frame

            raw_name = dataset.details.get("name", f"openml_{dataset_id}")
            df_name = sanitize_filename(f"openml_{raw_name}")

            file_path = RAW_DATA_DIR / f"{df_name}.csv"
            df.to_csv(file_path, index=False)

            success_count += 1
            print(f"    [OK] {df_name}: {len(df)} rows")

        except Exception as e:
            print(
                f"    [ERROR] Failed to process dataset {dataset_id}: "
                f"{str(e)}")

    print(f"OpenML done! Success: {success_count}/{len(dataset_ids)}")
    return success_count


def process_uciml_datasets() -> int:
    """Downloads and saves datasets from UCI. Returns the count of
    successful downloads."""
    print("Downloading datasets from UCI...")
    dataset_ids = get_ids_from_file(IDS_DIR / "uciml.txt")
    success_count = 0

    for dataset_id in dataset_ids:
        try:
            print(f"  Processing UCI ID: {dataset_id}")
            dataset = fetch_ucirepo(id=dataset_id)
            df = dataset.data.original

            raw_name = dataset.metadata.get("name", f"uci_{dataset_id}")
            df_name = sanitize_filename(f"uci_{raw_name}")

            file_path = RAW_DATA_DIR / f"{df_name}.csv"
            df.to_csv(file_path, index=False)

            success_count += 1
            print(f"    [OK] {df_name}: {len(df)} rows")

        except Exception as e:
            print(
                f"    [ERROR] Failed to process dataset {dataset_id}: "
                f"{str(e)}")

    print(f"UCI done! Success: {success_count}/{len(dataset_ids)}")
    return success_count


def main():
    print("Starting dataset download...\n")

    # Ensure the target directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    openml_success = process_openml_datasets()
    print()
    uci_success = process_uciml_datasets()

    total_success = openml_success + uci_success
    if total_success > 0:
        print(f"\nTotal datasets downloaded: {total_success}")
        print(f"Saved to: {RAW_DATA_DIR.relative_to(PROJECT_ROOT)}")
    else:
        print("\n[WARNING] No datasets were downloaded. Check ID files.")


if __name__ == "__main__":
    main()
