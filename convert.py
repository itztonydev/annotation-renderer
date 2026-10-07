from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


# Yellow in OpenCV BGR format.
LABEL_COLOR = (0, 255, 255)

POLYGON_THICKNESS = 2
BOX_THICKNESS = 2
POLYGON_ALPHA = 0.15


def find_annotations(
    dataset_dir: Path,
) -> Path:
    """
    Find annotations.xml anywhere inside the extracted dataset.
    """
    matches = list(
        dataset_dir.rglob("annotations.xml")
    )

    if not matches:
        raise FileNotFoundError(
            f"annotations.xml not found in {dataset_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple annotations.xml files found in "
            f"{dataset_dir}: {matches}"
        )

    return matches[0]


def find_image(
    dataset_dir: Path,
    image_name: str,
) -> Path | None:
    """
    Find the image referenced by the CVAT XML.

    CVAT commonly stores names such as:

        images/image001.jpg

    but the archive may have a slightly different
    directory structure, so we try both the relative
    path and the basename.
    """
    image_name = image_name.replace(
        "\\",
        "/",
    )

    # First: exact path relative to extracted dataset.
    direct_path = (
        dataset_dir / image_name
    )

    if direct_path.is_file():
        return direct_path

    # Second: search recursively by filename.
    filename = Path(image_name).name

    matches = list(
        dataset_dir.rglob(filename)
    )

    if not matches:
        return None

    if len(matches) > 1:
        # Prefer an image under an "images" directory.
        image_matches = [
            path
            for path in matches
            if "images" in path.parts
        ]

        if len(image_matches) == 1:
            return image_matches[0]

        # Otherwise use the first deterministic result.
        return sorted(matches)[0]

    return matches[0]


def parse_points(
    points_string: str,
) -> np.ndarray:
    """
    Convert:

        100,200;150,250;200,200

    into an OpenCV integer point array.
    """
    points = []

    for point in points_string.split(";"):
        point = point.strip()

        if not point:
            continue

        x, y = point.split(",")

        points.append(
            [
                int(float(x)),
                int(float(y)),
            ]
        )

    if len(points) < 3:
        raise ValueError(
            f"Polygon must have at least 3 points: "
            f"{points_string}"
        )

    return np.asarray(
        points,
        dtype=np.int32,
    )


def draw_polygon(
    image: np.ndarray,
    polygon: ET.Element,
) -> np.ndarray:
    """Draw one polygon with transparent fill and border."""
    points_string = polygon.get("points")

    if not points_string:
        return image

    points = parse_points(
        points_string
    )

    # Transparent fill.
    overlay = image.copy()

    # cv2.fillPoly(
    #     overlay,
    #     [points],
    #     LABEL_COLOR,
    # )

    image = cv2.addWeighted(
        overlay,
        POLYGON_ALPHA,
        image,
        1.0 - POLYGON_ALPHA,
        0,
    )

    # Polygon border.
    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=LABEL_COLOR,
        thickness=POLYGON_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    return image


def draw_box(
    image: np.ndarray,
    box: ET.Element,
) -> np.ndarray:
    """Draw one CVAT bounding box."""
    xtl = int(float(box.get("xtl", 0)))
    ytl = int(float(box.get("ytl", 0)))
    xbr = int(float(box.get("xbr", 0)))
    ybr = int(float(box.get("ybr", 0)))

    cv2.rectangle(
        image,
        (xtl, ytl),
        (xbr, ybr),
        LABEL_COLOR,
        BOX_THICKNESS,
    )

    return image


def annotate_image(
    image: np.ndarray,
    image_element: ET.Element,
) -> np.ndarray:
    """Apply all polygons and boxes for one image."""
    for polygon in image_element.findall(
        "polygon"
    ):
        image = draw_polygon(
            image,
            polygon,
        )

    for box in image_element.findall(
        "box"
    ):
        image = draw_box(
            image,
            box,
        )

    return image


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
) -> tuple[int, int]:
    """
    Convert one extracted dataset.

    Returns:
        (processed_count, skipped_count)
    """
    annotations_path = find_annotations(
        dataset_dir
    )

    print(
        f"Annotations: {annotations_path}"
    )

    tree = ET.parse(
        annotations_path
    )

    root = tree.getroot()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0
    skipped = 0

    for image_element in root.findall(
        ".//image"
    ):
        image_name = image_element.get(
            "name"
        )

        if not image_name:
            print(
                "  WARNING: image has no name"
            )
            skipped += 1
            continue

        image_path = find_image(
            dataset_dir,
            image_name,
        )

        if image_path is None:
            print(
                f"  WARNING: image not found: "
                f"{image_name}"
            )
            skipped += 1
            continue

        print(
            f"  Processing: {image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"  WARNING: could not read: "
                f"{image_path}"
            )
            skipped += 1
            continue

        image = annotate_image(
            image,
            image_element,
        )

        # Always produce PNG.
        output_name = (
            Path(image_path.name).stem
            + ".png"
        )

        output_path = (
            output_dir / output_name
        )

        # Prevent accidental overwrite if two XML
        # entries resolve to the same filename.
        if output_path.exists():
            stem = output_path.stem
            suffix = output_path.suffix

            counter = 2

            while output_path.exists():
                output_path = (
                    output_dir
                    / f"{stem}_{counter}{suffix}"
                )
                counter += 1

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:
            print(
                f"  WARNING: could not write: "
                f"{output_path}"
            )
            skipped += 1
            continue

        processed += 1

        print(
            f"       -> {output_path}"
        )

    return processed, skipped
