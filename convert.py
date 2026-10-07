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
    Load condition -> RGB hex colors from YAML.

    YAML:
        Whole: "#0000FF"

    OpenCV internally uses BGR, so colors are
    converted from RGB -> BGR here.
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
            "Color configuration must be a YAML mapping."
        )

    colors = {}

    for condition, hex_color in config.items():
        if not isinstance(condition, str):
            raise ValueError(
                f"Invalid condition: {condition!r}"
            )

        if not isinstance(hex_color, str):
            raise ValueError(
                f"Color for '{condition}' must be "
                f"a hex string."
            )

        value = hex_color.strip()

        if value.startswith("#"):
            value = value[1:]

        if len(value) != 6:
            raise ValueError(
                f"Invalid color for '{condition}': "
                f"#{value}. Expected #RRGGBB."
            )

        try:
            red = int(value[0:2], 16)
            green = int(value[2:4], 16)
            blue = int(value[4:6], 16)
        except ValueError as exc:
            raise ValueError(
                f"Invalid hex color for "
                f"'{condition}': #{value}"
            ) from exc

        # RGB -> OpenCV BGR
        colors[condition] = (
            blue,
            green,
            red,
        )

    return colors


def get_condition(
    annotation: ET.Element,
) -> str:
    """
    Extract the Condition attribute from an
    annotation.

    Example XML:

        <polygon label="Rice">
            <attribute name="Condition">
                Whole
            </attribute>
        </polygon>
    """
    for attribute in annotation.findall(
        "attribute"
    ):
        name = attribute.get("name")

        if name == "Condition":
            value = attribute.text

            if value is None:
                raise ValueError(
                    "Condition attribute is empty."
                )

            value = value.strip()

            if not value:
                raise ValueError(
                    "Condition attribute is empty."
                )

            return value

    raise ValueError(
        f"Annotation does not contain a "
        f"'Condition' attribute. "
        f"Label: {annotation.get('label')!r}"
    )


def get_annotation_color(
    annotation: ET.Element,
    colors: dict[str, tuple[int, int, int]],
) -> tuple[int, int, int]:
    """
    Get annotation color based on Condition,
    NOT the object label.
    """
    condition = get_condition(
        annotation
    )

    if condition not in colors:
        available = ", ".join(
            sorted(colors.keys())
        )

        raise ValueError(
            f"Unknown condition: "
            f"'{condition}'. "
            f"Available conditions: {available}"
        )

    return colors[condition]


def find_annotations(
    dataset_dir: Path,
) -> Path:
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
    colors: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """
    Draw polygon using its Condition color.
    """
    points_string = polygon.get("points")

    if not points_string:
        return image

    color = get_annotation_color(
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

    # Colored border.
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
    """
    Draw bounding box using its Condition color.
    """
    color = get_annotation_color(
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
    """
    Apply all polygon and box annotations.

    Color is determined by:
        annotation -> Condition -> colors.yaml
    """
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
                "WARNING: image has no name"
            )
            skipped += 1
            continue

        image_path = find_image(
            dataset_dir,
            image_name,
        )

        if image_path is None:
            print(
                f"WARNING: image not found: "
                f"{image_name}"
            )
            skipped += 1
            continue

        print(
            f"Processing: {image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"WARNING: could not read: "
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
                f"WARNING: could not write: "
                f"{output_path}"
            )
            skipped += 1
            continue

        processed += 1

        print(
            f"  -> {output_path}"
        )

    return processed, skipped
