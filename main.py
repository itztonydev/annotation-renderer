from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convert import convert_dataset
from extract import extract_input_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ZIP/TAR datasets and convert "
            "their CVAT annotations into annotated PNG images."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Directory containing ZIP/TAR files.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Directory where processed datasets will be created.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input
    output_dir = args.output

    if not input_dir.exists():
        print(
            f"Input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 1

    if not input_dir.is_dir():
        print(
            f"Input is not a directory: {input_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # ---------------------------------------------------------
    # Extract all archives
    # ---------------------------------------------------------
    try:
        datasets = extract_input_directory(
            input_dir,
            output_dir,
        )
    except Exception as exc:
        print(
            f"Extraction failed: {exc}",
            file=sys.stderr,
        )
        return 1

    if not datasets:
        print("No ZIP/TAR archives found.")
        return 0

    print()

    # ---------------------------------------------------------
    # Convert each extracted dataset
    # ---------------------------------------------------------
    failed = 0

    for dataset_dir in datasets:
        print("=" * 60)
        print(f"Dataset: {dataset_dir.name}")
        print("=" * 60)

        # Keep extracted files in a temporary subdirectory,
        # and put the final annotated images in "annotated".
        annotated_dir = dataset_dir / "annotated"

        try:
            convert_dataset(
                dataset_dir,
                annotated_dir,
            )

        except Exception as exc:
            failed += 1

            print(
                f"Failed: {dataset_dir}"
            )
            print(
                f"Reason: {exc}",
                file=sys.stderr,
            )

        print()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("=" * 60)
    print("Finished.")
    print(f"Datasets: {len(datasets)}")
    print(f"Failed  : {failed}")
    print("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

