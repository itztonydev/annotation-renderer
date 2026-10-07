from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


LABEL_COLOR = (0, 255, 255)  # Yellow


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def find_annotations_file(dataset_dir: Path) -> Path:
    """
    Find annotations.xml inside the extracted dataset.
    """
    candidates = list(dataset_dir.rglob("annotations.xml"))

    if not candidates:
        raise FileNotFoundError(
            f"annotations.xml not found in {dataset_dir}"
        )

    return candidates[0]


def find_image(
    dataset_dir: Path,
    image_name: str,
) -> Path | None:
    """
    Locate an image referenced by annotations.xml.

    First tries the path directly, then searches recursively
    by filename.
    """
    image_name = image_name.replace("\\", "/")

    # Try path relative to dataset root.
    direct_path = dataset_dir / image_name

    if direct_path.exists():
        return direct_path

    # Try only the basename.
    filename = Path(image_name).name

    matches = list(dataset_dir.rglob(filename))

    if matches:
        return matches[0]

    return None


def parse_points(points_string: str) -> np.ndarray:
    """
    Convert CVAT-style polygon points:

        "100,200;150,250;200,200"

    into an OpenCV array.
    """
    points = []

    for point in points_string.split(";"):
        x, y = point.split(",")

        points.append(
            [
                int(float(x)),
                int(float(y)),
            ]
        )

    return np.array(
        points,
        dtype=np.int32,
    )


def draw_polygon(
    img: np.ndarray,
    polygon: ET.Element,
) -> np.ndarray:
    points_string = polygon.get("points")

    if not points_string:
        return img

    points_array = parse_points(points_string)

    # Transparent fill.
    overlay = img.copy()

    cv2.fillPoly(
        overlay,
        [points_array],
        LABEL_COLOR,
    )

    img = cv2.addWeighted(
        overlay,
        0.15,
        img,
        0.85,
        0,
    )

    # Polygon border.
    cv2.polylines(
        img,
        [points_array],
        isClosed=True,
        color=LABEL_COLOR,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return img


def draw_box(
    img: np.ndarray,
    box: ET.Element,
) -> np.ndarray:
    xtl = int(float(box.get("xtl")))
    ytl = int(float(box.get("ytl")))
    xbr = int(float(box.get("xbr")))
    ybr = int(float(box.get("ybr")))

    cv2.rectangle(
        img,
        (xtl, ytl),
        (xbr, ybr),
        LABEL_COLOR,
        2,
    )

    return img


def annotate_image(
    img: np.ndarray,
    image_element: ET.Element,
) -> np.ndarray:
    """
    Draw all polygons and bounding boxes belonging
    to one CVAT image element.
    """
    for polygon in image_element.findall("polygon"):
        img = draw_polygon(
            img,
            polygon,
        )

    for box in image_element.findall("box"):
        img = draw_box(
            img,
            box,
        )

    return img


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
) -> None:
    """
    Convert one extracted dataset.

    The annotated images are written to output_dir.
    """
    annotations_file = find_annotations_file(dataset_dir)

    print(f"Annotations: {annotations_file}")

    tree = ET.parse(annotations_file)
    root = tree.getroot()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0
    skipped = 0

    for image_element in root.findall(".//image"):
        image_name = image_element.get("name")

        if not image_name:
            print("Skipping image without name")
            skipped += 1
            continue

        image_path = find_image(
            dataset_dir,
            image_name,
        )

        if image_path is None:
            print(
                f"Image not found: {image_name}"
            )
            skipped += 1
            continue

        print(f"Processing: {image_path.name}")

        img = cv2.imread(
            str(image_path)
        )

        if img is None:
            print(
                f"Could not read: {image_path}"
            )
            skipped += 1
            continue

        img = annotate_image(
            img,
            image_element,
        )

        # Preserve original filename but output as PNG.
        output_name = (
            Path(image_path.name).stem
            + ".png"
        )

        output_path = output_dir / output_name

        success = cv2.imwrite(
            str(output_path),
            img,
        )

        if not success:
            print(
                f"Could not write: {output_path}"
            )
            skipped += 1
            continue

        processed += 1

        print(
            f"  -> {output_path}"
        )

    print(
        f"Completed: {processed} images, "
        f"{skipped} skipped"
    )
