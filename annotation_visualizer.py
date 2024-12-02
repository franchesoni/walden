from pathlib import Path 
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Define constants
TILE_SIZE = 256          # Size of each tile in pixels
CROP_SIZE = 512          # Desired crop size (512x512)
HALF_CROP = CROP_SIZE // 2  # Half of the crop size
DOT_RADIUS = 5           # Radius of the red dot
SQUARE_SIZE = 10         # Size of the red square (if preferred)

# File paths and image name
filepath = "annotations/Image_05.vsi - 40x_BF_Z_01-points.tsv"
imgname = "img5"
tiles_dir = Path(f"dataset/{imgname}/tiles")

# Load the annotations
annotations = pd.read_csv(filepath, sep="\t")

# Load a font for text overlay (optional: specify a TTF font file if desired)
try:
    font = ImageFont.truetype("arial.ttf", size=16)
except IOError:
    font = ImageFont.load_default()

# Iterate through each annotation
for index, row in annotations.iterrows():
    x = row['x']  # Column (horizontal)
    y = row['y']  # Row (vertical)
    class_label = row.get('class', 'Unknown')  # Get the class label, default to 'Unknown'

    # Calculate the starting coordinates by flooring to the nearest multiple of TILE_SIZE
    xstart = (x // TILE_SIZE) * TILE_SIZE
    ystart = (y // TILE_SIZE) * TILE_SIZE

    # Initialize a blank composite image (3x3 tiles)
    composite_image = Image.new('RGB', (TILE_SIZE * 3, TILE_SIZE * 3), (255, 255, 255))

    # Iterate through surrounding tiles (rows: i, columns: j)
    for i, dy in enumerate([-TILE_SIZE, 0, TILE_SIZE]):  # Vertical movement (rows)
        for j, dx in enumerate([-TILE_SIZE, 0, TILE_SIZE]):  # Horizontal movement (columns)
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
                    if tile_image.mode != 'RGB':
                        tile_image = tile_image.convert('RGB')
                    # Paste the tile into the composite at the correct position
                    # (j corresponds to columns, i corresponds to rows)
                    composite_image.paste(tile_image, (j * TILE_SIZE, i * TILE_SIZE))
                except Exception as e:
                    print(f"Error loading tile {tile_name}: {e}")
            else:
                # Optional: Indicate missing tiles with a placeholder
                missing_tile = Image.new('RGB', (TILE_SIZE, TILE_SIZE), (200, 200, 200))
                draw_missing = ImageDraw.Draw(missing_tile)
                draw_missing.line((0, 0) + missing_tile.size, fill=(150, 150, 150), width=3)
                draw_missing.line((0, missing_tile.size[1], missing_tile.size[0], 0), fill=(150, 150, 150), width=3)
                composite_image.paste(missing_tile, (j * TILE_SIZE, i * TILE_SIZE))

    # Calculate the annotation's position within the composite image
    # The composite image spans from (xstart - TILE_SIZE, ystart - TILE_SIZE) to (xstart + 2*TILE_SIZE, ystart + 2*TILE_SIZE)
    # Therefore, the annotation position relative to composite image:
    rel_x = x - (xstart - TILE_SIZE)
    rel_y = y - (ystart - TILE_SIZE)

    # Define the crop box centered around the annotation position
    left = rel_x - HALF_CROP
    upper = rel_y - HALF_CROP
    right = rel_x + HALF_CROP
    lower = rel_y + HALF_CROP

    # Ensure the crop box is within the bounds of the composite image
    # If not, adjust the box and pad the image accordingly
    pad_left = max(0, -left)
    pad_upper = max(0, -upper)
    pad_right = max(0, lower - composite_image.height)
    pad_lower = max(0, right - composite_image.width)

    # Adjust the crop box to be within the image
    left = max(0, left)
    upper = max(0, upper)
    right = min(composite_image.width, right)
    lower = min(composite_image.height, lower)

    # Crop the composite image
    cropped_image = composite_image.crop((left, upper, right, lower))

    # If padding is needed, create a new white image and paste the cropped image onto it
    if any([pad_left, pad_upper, pad_right, pad_lower]):
        new_width = right - left + pad_left + pad_right
        new_height = lower - upper + pad_upper + pad_lower
        padded_image = Image.new('RGB', (CROP_SIZE, CROP_SIZE), (255, 255, 255))
        padded_image.paste(cropped_image, (pad_left, pad_upper))
        cropped_image = padded_image

    # Overlay the class label on the cropped image
    draw = ImageDraw.Draw(cropped_image)
    text_position = (10, 10)  # Position of the text
    text = f"Class: {class_label}"
    text_color = (255, 0, 0)  # Red color for text
    draw.text(text_position, text, fill=text_color, font=font)

    # Add a red dot or square at the annotation position
    annotation_position = (HALF_CROP, HALF_CROP)  # Center of the cropped image

    # Option 1: Red dot
    draw.ellipse(
        (annotation_position[0] - DOT_RADIUS, annotation_position[1] - DOT_RADIUS, 
         annotation_position[0] + DOT_RADIUS, annotation_position[1] + DOT_RADIUS), 
        fill=(255, 0, 0), outline=(255, 0, 0)
    )

    # Option 2: Red square (uncomment if preferred)
    # draw.rectangle(
    #     (annotation_position[0] - SQUARE_SIZE // 2, annotation_position[1] - SQUARE_SIZE // 2, 
    #      annotation_position[0] + SQUARE_SIZE // 2, annotation_position[1] + SQUARE_SIZE // 2), 
    #     fill=(255, 0, 0), outline=(255, 0, 0)
    # )

    # Define the output filename (unique for each annotation)
    output_filename = f"ann.png"

    # Save the cropped image with annotations
    cropped_image.save(output_filename)
    print(f"Saved cropped image '{output_filename}' for annotation at x: {x}, y: {y} with class: {class_label}")

    # Manual control of the loop
    input("Press Enter to continue to the next annotation...")
