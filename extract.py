from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def is_supported_archive(path: Path) -> bool:
    """Check whether a file is a supported ZIP or TAR archive."""
    return (
        path.is_file()
        and (
            zipfile.is_zipfile(path)
            or tarfile.is_tarfile(path)
        )
    )


def _safe_path(base_dir: Path, member_name: str) -> Path:
    """
    Prevent archive path traversal such as:
        ../../some_file
    """
    base_dir = base_dir.resolve()
    target = (base_dir / member_name).resolve()

    try:
        target.relative_to(base_dir)
    except ValueError:
        raise RuntimeError(
            f"Unsafe path in archive: {member_name}"
        )

    return target


def extract_zip(
    archive_path: Path,
    output_dir: Path,
) -> None:
    """Safely extract a ZIP archive."""
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            _safe_path(
                output_dir,
                member.filename,
            )

        archive.extractall(output_dir)


def extract_tar(
    archive_path: Path,
    output_dir: Path,
) -> None:
    """Safely extract a TAR archive."""
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            _safe_path(
                output_dir,
                member.name,
            )

        archive.extractall(output_dir)


def extract_archive(
    archive_path: Path,
    output_dir: Path,
) -> None:
    """
    Extract either ZIP or TAR archive into output_dir.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if zipfile.is_zipfile(archive_path):
        extract_zip(
            archive_path,
            output_dir,
        )
        return

    if tarfile.is_tarfile(archive_path):
        extract_tar(
            archive_path,
            output_dir,
        )
        return

    raise ValueError(
        f"Unsupported archive: {archive_path}"
    )


def find_archives(
    input_dir: Path,
) -> list[Path]:
    """
    Find ZIP/TAR files directly inside INPUT.
    """
    archives = []

    for path in sorted(input_dir.iterdir()):
        if is_supported_archive(path):
            archives.append(path)

    return archives
