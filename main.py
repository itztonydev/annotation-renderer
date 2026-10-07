from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from convert import convert_dataset
from extract import extract_archive, find_archives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ZIP/TAR datasets and create "
            "annotated PNG images."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Input directory containing .zip/.tar files."
        ),
    )

    parser.add_argument(
        "output",
        type=Path,
        help=(
            "Output directory for annotated images."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    # ---------------------------------------------------------
    # Validate INPUT
    # ---------------------------------------------------------
    if not input_dir.exists():
        print(
            f"ERROR: input directory does not exist: "
            f"{input_dir}",
            file=sys.stderr,
        )
        return 1

    if not input_dir.is_dir():
        print(
            f"ERROR: input is not a directory: "
            f"{input_dir}",
            file=sys.stderr,
        )
        return 1

    # ---------------------------------------------------------
    # Create OUTPUT
    # ---------------------------------------------------------
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Find archives
    # ---------------------------------------------------------
    archives = find_archives(
        input_dir
    )

    if not archives:
        print(
            "No ZIP/TAR archives found."
        )
        return 0

    print(
        f"Found {len(archives)} archive(s)."
    )
    print()

    total_processed = 0
    total_skipped = 0
    failed = 0

    # ---------------------------------------------------------
    # Process every archive independently
    # ---------------------------------------------------------
    for archive_path in archives:
        dataset_name = archive_path.name

        # Remove archive extension.
        if dataset_name.endswith(
            ".tar.gz"
        ):
            dataset_name = dataset_name[
                :-7
            ]

        elif dataset_name.endswith(
            ".tar.bz2"
        ):
            dataset_name = dataset_name[
                :-8
            ]

        elif dataset_name.endswith(
            ".tar.xz"
        ):
            dataset_name = dataset_name[
                :-7
            ]

        elif dataset_name.endswith(
            ".tgz"
        ):
            dataset_name = dataset_name[
                :-4
            ]

        else:
            dataset_name = archive_path.stem

        final_output_dir = (
            output_dir / dataset_name
        )

        print("=" * 60)
        print(
            f"Archive: {archive_path.name}"
        )
        print(
            f"Output : {final_output_dir}"
        )
        print("=" * 60)

        try:
            # Temporary extraction directory.
            with tempfile.TemporaryDirectory(
                prefix="annotation_"
            ) as temp_dir:
                temp_dir = Path(temp_dir)

                print(
                    f"Extracting to temporary directory..."
                )

                extract_archive(
                    archive_path,
                    temp_dir,
                )

                # If this output already exists,
                # replace it with the new result.
                if final_output_dir.exists():
                    shutil.rmtree(
                        final_output_dir
                    )

                final_output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                processed, skipped = (
                    convert_dataset(
                        temp_dir,
                        final_output_dir,
                    )
                )

                total_processed += processed
                total_skipped += skipped

                print()
                print(
                    f"Processed: {processed}"
                )
                print(
                    f"Skipped  : {skipped}"
                )

        except Exception as exc:
            failed += 1

            # Remove incomplete output.
            if final_output_dir.exists():
                shutil.rmtree(
                    final_output_dir,
                    ignore_errors=True,
                )

            print(
                f"ERROR: failed to process "
                f"{archive_path.name}: {exc}",
                file=sys.stderr,
            )

        print()

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------
    print("=" * 60)
    print("Finished")
    print("=" * 60)
    print(
        f"Archives found : {len(archives)}"
    )
    print(
        f"Images created : {total_processed}"
    )
    print(
        f"Images skipped : {total_skipped}"
    )
    print(
        f"Archives failed: {failed}"
    )
    print(
        f"Output         : {output_dir}"
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
