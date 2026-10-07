from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml


POLYGON_THICKNESS = 2
BOX_THICKNESS = 2
POLYGON_ALPHA = 0.15


def load_colors(
    colors_file: Path,
) -> dict[str, tuple[int, int, int]]:
    """
    Load label colors from YAML.

    YAML uses normal RGB hex colors:

        Whole: "#0000FF"

    OpenCV uses BGR, so the values are converted here.
    """
    if not colors_file.is_file():
        raise FileNotFoundError(
            f"Color configuration not found: "
            f"{colors_file}"
        )

    with colors_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Color configuration must contain "
            f"a YAML mapping: {colors_file}"
        )

    colors = {}

    for label, hex_color in config.items():
        if not isinstance(label, str):
            raise ValueError(
                f"Invalid label in {colors_file}: "
                f"{label!r}"
            )

        if not isinstance(hex_color, str):
            raise ValueError(
                f"Color for '{label}' must be a "
                f"hex string, got {hex_color!r}"
            )

        hex_color = hex_color.strip()

        if hex_color.startswith("#"):
            hex_color = hex_color[1:]

        if len(hex_color) != 6:
            raise ValueError(
                f"Invalid hex color for '{label}': "
                f"#{hex_color}. Expected #RRGGBB."
            )

        try:
            red = int(hex_color[0:2], 16)
            green = int(hex_color[2:4], 16)
            blue = int(hex_color[4:6], 16)
        except ValueError as exc:
            raise ValueError(
                f"Invalid hex color for '{label}': "
                f"#{hex_color}"
            ) from exc

        # OpenCV uses BGR rather than RGB.
        colors[label] = (
            blue,
            green,
            red,
        )

    return colors


def find_annotations(
    dataset_dir: Path,
) -> Path:
    """Find annotations.xml inside the dataset."""
    matches = list(
        dataset_dir.rglob("annotations.xml")
    )

    if not matches:
        raise FileNotFoundError(
            f"annotations.xml not found in "
            f"{dataset_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple annotations.xml files found "
            f"in {dataset_dir}: {matches}"
        )

    return matches[0]


def find_image(
    dataset_dir: Path,
    image_name: str,
) -> Path | None:
    """
    Find the image referenced by CVAT XML.

    Exact path is preferred. If that doesn't exist,
    search by filename.
    """
    image_name = image_name.replace(
        "\\",
        "/",
    )

    direct_path = (
        dataset_dir / image_name
    )

    if direct_path.is_file():
        return direct_path

    filename = Path(image_name).name

    matches = sorted(
        dataset_dir.rglob(filename)
    )

    if not matches:
        return None

    # If there are multiple files with the same
    # basename, don't guess.
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous image reference "
            f"'{image_name}'. Found multiple files:\n"
            + "\n".join(
                f"  {path}"
                for path in matches
            )
        )

    return matches[0]


def parse_points(
    points_string: str,
) -> np.ndarray:
    """Convert CVAT points into an OpenCV array."""
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
            f"Polygon must contain at least "
            f"3 points: {points_string}"
        )

    return np.asarray(
        points,
        dtype=np.int32,
    )


def get_label_color(
    element: ET.Element,
    colors: dict[str, tuple[int, int, int]],
) -> tuple[int, int, int]:
    """
    Get the color for an annotation element.

    The label comes from the CVAT XML:
        <polygon label="Whole" ...>

    or:
        <box label="Broken" ...>
    """
    label = element.get("label")

    if not label:
        raise ValueError(
            "Annotation has no 'label' attribute."
        )

    if label not in colors:
        available = ", ".join(
            sorted(colors.keys())
        )

        raise ValueError(
            f"Unknown annotation label: "
            f"'{label}'. "
            f"Available labels: {available}"
        )

    return colors[label]


def draw_polygon(
    image: np.ndarray,
    polygon: ET.Element,
    colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """Draw one colored polygon."""
    points_string = polygon.get("points")

    if not points_string:
        return image

    color = get_label_color(
        polygon,
        colors,
    )

    points = parse_points(
        points_string
    )

    # Transparent fill.
    overlay = image.copy()

    cv2.fillPoly(
        overlay,
        [points],
        color,
    )

    image = cv2.addWeighted(
        overlay,
        POLYGON_ALPHA,
        image,
        1.0 - POLYGON_ALPHA,
        0,
    )

    # Solid border.
    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=color,
        thickness=POLYGON_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    return image


def draw_box(
    image: np.ndarray,
    box: ET.Element,
    colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """Draw one colored bounding box."""
    color = get_label_color(
        box,
        colors,
    )

    xtl = int(
        float(box.get("xtl", 0))
    )
    ytl = int(
        float(box.get("ytl", 0))
    )
    xbr = int(
        float(box.get("xbr", 0))
    )
    ybr = int(
        float(box.get("ybr", 0))
    )

    cv2.rectangle(
        image,
        (xtl, ytl),
        (xbr, ybr),
        color,
        BOX_THICKNESS,
    )

    return image


def annotate_image(
    image: np.ndarray,
    image_element: ET.Element,
    colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """Apply all annotations belonging to one image."""
    for polygon in image_element.findall(
        "polygon"
    ):
        image = draw_polygon(
            image,
            polygon,
            colors,
        )

    for box in image_element.findall(
        "box"
    ):
        image = draw_box(
            image,
            box,
            colors,
        )

    return image


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
    colors_file: Path,
) -> tuple[int, int]:
    """
    Convert one extracted dataset.

    Returns:
        (processed_count, skipped_count)
    """
    annotations_path = find_annotations(
        dataset_dir
    )

    colors = load_colors(
        colors_file
    )

    print(
        f"Annotations: {annotations_path}"
    )

    print(
        f"Colors     : {colors_file}"
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

        try:
            image_path = find_image(
                dataset_dir,
                image_name,
            )
        except RuntimeError:
            raise

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
            colors,
        )

        output_name = (
            Path(image_path.name).stem
            + ".png"
        )

        output_path = (
            output_dir / output_name
        )

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
