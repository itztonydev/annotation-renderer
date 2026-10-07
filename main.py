
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np


XML_FILE = "annotations.xml"
IMAGE_DIR = "images"
OUTPUT_DIR = "annotated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tree = ET.parse(XML_FILE)
root = tree.getroot()


# ---------------------------------------------------------
# Colors
# ---------------------------------------------------------

LABEL_COLOR = (0, 255, 255)  # Yellow


# ---------------------------------------------------------
# Process images
# ---------------------------------------------------------

for image_element in root.findall(".//image"):

    image_name = image_element.get("name")

    filename = os.path.basename(image_name)

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not read: {image_path}")
        continue

    print(f"Processing: {filename}")


    # =====================================================
    # POLYGONS
    # =====================================================

    for polygon in image_element.findall("polygon"):

        points_string = polygon.get("points")

        points = []

        for point in points_string.split(";"):

            x, y = point.split(",")

            points.append([
                int(float(x)),
                int(float(y))
            ])

        points_array = np.array(
            points,
            dtype="int32"
        )


        # -------------------------------------------------
        # Draw polygon border
        # -------------------------------------------------

        cv2.polylines(
            img,
            [points_array],
            isClosed=True,
            color=LABEL_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA
        )


        # -------------------------------------------------
        # Transparent polygon fill
        # -------------------------------------------------

        overlay = img.copy()

        # cv2.fillPoly(
        #     overlay,
        #     [points_array],
        #     LABEL_COLOR
        # )

        img = cv2.addWeighted(
            overlay,
            0.15,
            img,
            0.85,
            0
        )


    # =====================================================
    # BOUNDING BOXES
    # =====================================================

    for box in image_element.findall("box"):

        xtl = int(float(box.get("xtl")))
        ytl = int(float(box.get("ytl")))
        xbr = int(float(box.get("xbr")))
        ybr = int(float(box.get("ybr")))


        # -------------------------------------------------
        # Draw bounding box
        # -------------------------------------------------

        cv2.rectangle(
            img,
            (xtl, ytl),
            (xbr, ybr),
            LABEL_COLOR,
            2
        )


    # =====================================================
    # SAVE PNG
    # =====================================================

    output_name = (
        os.path.splitext(filename)[0]
        + ".png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    cv2.imwrite(
        output_path,
        img
    )

    print(f"  -> {output_path}")


print("\nFinished.")
