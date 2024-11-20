import os
import shutil
from pathlib import Path
from fire import Fire
from PIL import Image
import math
import warnings

# Increase the decompression bomb limit to handle large images
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def main(srcdir="out", reset=False):
    srcdir = Path(srcdir)
    dstdir = Path("overlapping_tiles")
    if dstdir.exists() and reset:
        shutil.rmtree(dstdir)
    dstdir.mkdir(exist_ok=True)

    # Extract dimensions of the slide
    with open(srcdir / "extracted_metadata.txt", "r") as f:
        lines = f.read().split("\n")
        width = int(lines[1].split(":")[1])
        height = int(lines[2].split(":")[1])
        best_focus = int(lines[4].split("[")[1].split("]")[0])
    print("Height:", height, "Width:", width)

    big_tile_size = 16384  # Size of the big tiles (2**14)
    small_tile_size = 1024  # Size of the small tiles
    small_tile_step = 512  # Step size for overlapping tiles

    # Precompute the list of small tile starting positions
    small_tile_col0_list = list(range(0, width - small_tile_size + 1, small_tile_step))
    small_tile_row0_list = list(range(0, height - small_tile_size + 1, small_tile_step))

    num_big_tiles_col = math.ceil(width / big_tile_size)
    num_big_tiles_row = math.ceil(height / big_tile_size)

    print(
        f"Number of big tiles - Columns: {num_big_tiles_col}, Rows: {num_big_tiles_row}"
    )

    # Loop over big tiles
    for btrow in range(num_big_tiles_row):
        for btcol in range(num_big_tiles_col):
            # **Important Change:** Swap btcol and btrow in the tile name to match {col}_{row} format
            big_tile_name = f"img_full_crop_z{best_focus}_tile_{btcol}_{btrow}.jpeg"
            big_tile_path = srcdir / big_tile_name

            if not big_tile_path.exists():
                print(f"Big tile {big_tile_name} does not exist. Skipping.")
                continue

            # Load the big tile once
            with Image.open(big_tile_path) as big_tile:
                big_tile_col0 = btcol * big_tile_size
                big_tile_row0 = btrow * big_tile_size
                big_tile_col1 = big_tile_col0 + big_tile.width
                big_tile_row1 = big_tile_row0 + big_tile.height

                print(
                    f"Processing big tile {big_tile_name} at position (Col: {big_tile_col0}, Row: {big_tile_row0})"
                )

                # Determine overlapping small tiles
                overlapping_col0 = [
                    col0
                    for col0 in small_tile_col0_list
                    if col0 + small_tile_size > big_tile_col0 and col0 < big_tile_col1
                ]
                overlapping_row0 = [
                    row0
                    for row0 in small_tile_row0_list
                    if row0 + small_tile_size > big_tile_row0 and row0 < big_tile_row1
                ]

                # Loop over overlapping small tiles
                for row0 in overlapping_row0:
                    for col0 in overlapping_col0:
                        print(f"Processing {row0,col0}", end="\r")
                        col1 = col0 + small_tile_size
                        row1 = row0 + small_tile_size

                        # Compute the overlapping region between the small tile and the big tile
                        overlap_col0 = max(col0, big_tile_col0)
                        overlap_row0 = max(row0, big_tile_row0)
                        overlap_col1 = min(col1, big_tile_col1)
                        overlap_row1 = min(row1, big_tile_row1)

                        # Coordinates in the small tile
                        small_tile_col_offset = overlap_col0 - col0
                        small_tile_row_offset = overlap_row0 - row0

                        # Coordinates in the big tile
                        big_tile_col_crop_start = overlap_col0 - big_tile_col0
                        big_tile_row_crop_start = overlap_row0 - big_tile_row0
                        big_tile_col_crop_end = overlap_col1 - big_tile_col0
                        big_tile_row_crop_end = overlap_row1 - big_tile_row0

                        # Initialize or load the small tile
                        tile_output_name = f"tile_{row0}_{col0}.jpeg"
                        tile_output_path = dstdir / tile_output_name

                        if tile_output_path.exists():
                            # Open the existing small tile to paste additional regions
                            with Image.open(tile_output_path) as small_tile:
                                pass  # Just to ensure the file can be opened
                            small_tile = Image.open(tile_output_path)
                        else:
                            # Create a new small tile
                            small_tile = Image.new(
                                "RGB", (small_tile_size, small_tile_size)
                            )

                        # Extract the overlapping region from the big tile
                        region = big_tile.crop(
                            (
                                big_tile_col_crop_start,
                                big_tile_row_crop_start,
                                big_tile_col_crop_end,
                                big_tile_row_crop_end,
                            )
                        )

                        # Paste the region into the small tile at the correct offset
                        small_tile.paste(
                            region, (small_tile_col_offset, small_tile_row_offset)
                        )

                        # Save the small tile
                        small_tile.save(tile_output_path)
                        small_tile.close()

            print(f"Finished processing big tile {big_tile_name}.")

    print("All small tiles have been processed successfully.")


if __name__ == "__main__":
    Fire(main)
