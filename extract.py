from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path


SUPPORTED_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}


def is_archive(path: Path) -> bool:
    """Return True if the path is a supported ZIP/TAR archive."""
    return path.is_file() and (
        path.suffix.lower() in SUPPORTED_EXTENSIONS
        or tarfile.is_tarfile(path)
        or zipfile.is_zipfile(path)
    )


def safe_extract_zip(archive: Path, output_dir: Path) -> None:
    """Extract ZIP while preventing path traversal."""
    output_dir = output_dir.resolve()

    with zipfile.ZipFile(archive, "r") as zip_ref:
        for member in zip_ref.infolist():
            target = (output_dir / member.filename).resolve()

            if not str(target).startswith(str(output_dir)):
                raise RuntimeError(
                    f"Unsafe path in archive: {member.filename}"
                )

        zip_ref.extractall(output_dir)


def safe_extract_tar(archive: Path, output_dir: Path) -> None:
    """Extract TAR while preventing path traversal."""
    output_dir = output_dir.resolve()

    with tarfile.open(archive, "r:*") as tar_ref:
        for member in tar_ref.getmembers():
            target = (output_dir / member.name).resolve()

            if not str(target).startswith(str(output_dir)):
                raise RuntimeError(
                    f"Unsafe path in archive: {member.name}"
                )

        tar_ref.extractall(output_dir)


def extract_archive(archive: Path, output_dir: Path) -> Path:
    """
    Extract one archive into output_dir.

    Returns the extraction directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        safe_extract_zip(archive, output_dir)

    elif tarfile.is_tarfile(archive):
        safe_extract_tar(archive, output_dir)

    else:
        raise ValueError(f"Unsupported archive: {archive}")

    return output_dir


def extract_input_directory(
    input_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """
    Extract every supported archive from input_dir.

    Each archive gets its own output directory named after
    the archive filename without its extension.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(
        path
        for path in input_dir.iterdir()
        if is_archive(path)
    )

    extracted_dirs = []

    for archive in archives:
        archive_name = archive.stem

        # Handle .tar.gz / .tar.bz2 / .tar.xz etc.
        if archive.name.endswith(".tar.gz"):
            archive_name = archive.name[:-7]
        elif archive.name.endswith(".tar.bz2"):
            archive_name = archive.name[:-8]
        elif archive.name.endswith(".tar.xz"):
            archive_name = archive.name[:-7]
        elif archive.name.endswith(".tgz"):
            archive_name = archive.name[:-4]

        destination = output_dir / archive_name

        if destination.exists():
            shutil.rmtree(destination)

        print(f"Extracting: {archive}")
        print(f"       -> {destination}")

        extract_archive(
            archive,
            destination,
        )

        extracted_dirs.append(destination)

    return extracted_dirs
