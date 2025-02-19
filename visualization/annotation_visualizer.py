from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from classify import load_bounding_boxes_csv
import tqdm

# Define constants
TILE_SIZE = 256  # Size of each tile in pixels
CROP_SIZE = 512  # Desired crop size (512x512)
HALF_CROP = CROP_SIZE // 2  # Half of the crop size
DOT_RADIUS = 5  # Radius of the red dot
SQUARE_SIZE = 10  # Size of the red square (if preferred)

# File paths and image name
filepath = "annotations/Image_05.vsi - 40x_BF_Z_01-points.tsv"
imgname = "img5"
tiles_dir = Path(f"dataset/{imgname}/tiles")
vis_bbox = True

# Load the annotations
annotations = pd.read_csv(filepath, sep="\t")

# Load a font for text overlay (optional: specify a TTF font file if desired)
try:
    font = ImageFont.truetype("arial.ttf", size=16)
except IOError:
    font = ImageFont.load_default()

if vis_bbox:
    csv_file = f"out/img5/masks/global_bboxes.txt"
    bboxes = load_bounding_boxes_csv(csv_file)
    npbboxes = np.array(bboxes)
    bbox_rows, bbox_cols, bbox_heights, bbox_widths = (
        npbboxes[:, 0],
        npbboxes[:, 1],
        npbboxes[:, 2],
        npbboxes[:, 3],
    )
    bbox_bottom_rows = bbox_rows + bbox_heights
    bbox_right_cols = bbox_cols + bbox_widths
    bbox_center_rows = bbox_rows + bbox_heights / 2
    bbox_center_cols = bbox_cols + bbox_widths / 2

# Iterate through each annotation
for index, row in annotations.iterrows():
    x = row["x"]  # Column (horizontal)
    y = row["y"]  # Row (vertical)
    class_label = row.get(
        "class", "Unknown"
    )  # Get the class label, default to 'Unknown'

    # Find matching bounding boxes for the annotation
    if vis_bbox:
        manhattan_distances = np.abs(bbox_center_rows - y) + np.abs(
            bbox_center_cols - x
        )
        box_corner_distances = (bbox_heights / 2) + (bbox_widths / 2)
        min_distance_index = np.argmin(manhattan_distances)
        if (
            manhattan_distances[min_distance_index]
            < box_corner_distances[min_distance_index]
        ):
            match_idx = min_distance_index
        else:
            match_idx = None

    # Calculate the starting coordinates by flooring to the nearest multiple of TILE_SIZE
    xstart = (x // TILE_SIZE) * TILE_SIZE
    ystart = (y // TILE_SIZE) * TILE_SIZE

    # Initialize a blank composite image (3x3 tiles)
    composite_image = Image.new("RGB", (TILE_SIZE * 3, TILE_SIZE * 3), (255, 255, 255))

    # Iterate through surrounding tiles (rows: i, columns: j)
    for i, dy in enumerate([-TILE_SIZE, 0, TILE_SIZE]):  # Vertical movement (rows)
        for j, dx in enumerate(
            [-TILE_SIZE, 0, TILE_SIZE]
        ):  # Horizontal movement (columns)
            # Calculate the starting coordinates for the surrounding tile
            tile_x = xstart + dx
            tile_y = ystart + dy

            # Define the surrounding tile name using row (y) and column (x)
            tile_name = f"tile_{int(tile_y)}_{int(tile_x)}.jpeg"
            tile_path = tiles_dir / tile_name

            # Check if the tile exists
            if tile_path.exists():
                try:
                    tile_image = Image.open(tile_path)
                    # Ensure tile is in RGB mode
                    if tile_image.mode != "RGB":
                        tile_image = tile_image.convert("RGB")
                    # Paste the tile into the composite at the correct position
                    # (j corresponds to columns, i corresponds to rows)
                    composite_image.paste(tile_image, (j * TILE_SIZE, i * TILE_SIZE))
                except Exception as e:
                    print(f"Error loading tile {tile_name}: {e}")
            else:
                # Optional: Indicate missing tiles with a placeholder
                missing_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200))
                draw_missing = ImageDraw.Draw(missing_tile)
                draw_missing.line(
                    (0, 0) + missing_tile.size, fill=(150, 150, 150), width=3
                )
                draw_missing.line(
                    (0, missing_tile.size[1], missing_tile.size[0], 0),
                    fill=(150, 150, 150),
                    width=3,
                )
                composite_image.paste(missing_tile, (j * TILE_SIZE, i * TILE_SIZE))

    # Calculate the annotation's position within the composite image
    rel_x = x - (xstart - TILE_SIZE)
    rel_y = y - (ystart - TILE_SIZE)

    # Define the crop box centered around the annotation position
    left = rel_x - HALF_CROP
    upper = rel_y - HALF_CROP
    right = rel_x + HALF_CROP
    lower = rel_y + HALF_CROP

    # Ensure the crop box is within the bounds of the composite image
    pad_left = max(0, -left)
    pad_upper = max(0, -upper)
    pad_right = max(0, lower - composite_image.height)
    pad_lower = max(0, right - composite_image.width)

    left = max(0, left)
    upper = max(0, upper)
    right = min(composite_image.width, right)
    lower = min(composite_image.height, lower)

    cropped_image = composite_image.crop((left, upper, right, lower))

    if any([pad_left, pad_upper, pad_right, pad_lower]):
        new_width = right - left + pad_left + pad_right
        new_height = lower - upper + pad_upper + pad_lower
        padded_image = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (255, 255, 255))
        padded_image.paste(cropped_image, (pad_left, pad_upper))
        cropped_image = padded_image

    # Overlay the class label on the cropped image
    draw = ImageDraw.Draw(cropped_image)
    text_position = (10, 10)
    text = f"Class: {class_label}"
    text_color = (255, 0, 0)
    draw.text(text_position, text, fill=text_color, font=font)

    # Add a red dot at the annotation position
    annotation_position = (HALF_CROP, HALF_CROP)
    draw.ellipse(
        (
            annotation_position[0] - DOT_RADIUS,
            annotation_position[1] - DOT_RADIUS,
            annotation_position[0] + DOT_RADIUS,
            annotation_position[1] + DOT_RADIUS,
        ),
        fill=(255, 0, 0),
        outline=(255, 0, 0),
    )

    # Draw matching bounding boxes on the cropped image
    if vis_bbox and match_idx:
        bbox_top = bbox_rows[match_idx] - ystart + TILE_SIZE
        bbox_left = bbox_cols[match_idx] - xstart + TILE_SIZE
        bbox_bottom = bbox_bottom_rows[match_idx] - ystart + TILE_SIZE
        bbox_right = bbox_right_cols[match_idx] - xstart + TILE_SIZE

        # Adjust bounding box to crop-relative coordinates
        bbox_top = bbox_top - upper
        bbox_left = bbox_left - left
        bbox_bottom = bbox_bottom - upper
        bbox_right = bbox_right - left

        draw.rectangle(
            [(bbox_left, bbox_top), (bbox_right, bbox_bottom)], outline="blue", width=3
        )

    # Save the cropped image with annotations
    output_filename = f"ann.png"
    cropped_image.save(output_filename)
    print(
        f"Saved cropped image '{output_filename}' for annotation at x: {x}, y: {y} with class: {class_label}"
    )

    # Manual control of the loop
    input("Press Enter to continue to the next annotation...")
