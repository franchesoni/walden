import shutil
from pathlib import Path
import subprocess
from fire import Fire
import re


def extract_largest_image_info(metadata_path="out/metadata.txt"):
    largest_width = 0
    largest_height = 0
    largest_series = None
    z_depths = 0

    with open(metadata_path, "r") as file:
        content = file.read()

    # Split content by series
    series_blocks = re.split(r"Series #\d+ :", content)

    for series_number, block in enumerate(series_blocks[1:], start=0):
        # Extract width, height, and Z depth
        width_match = re.search(r"Width = (\d+)", block)
        height_match = re.search(r"Height = (\d+)", block)
        size_z_match = re.search(r"SizeZ = (\d+)", block)

        if width_match and height_match:
            width = int(width_match.group(1))
            height = int(height_match.group(1))

            if width * height > largest_width * largest_height:
                largest_width = width
                largest_height = height
                largest_series = series_number
                z_depths = int(size_z_match.group(1)) if size_z_match else 0

    return largest_series, largest_width, largest_height, z_depths


def run_showinf(imgpath, output_path="out/metadata.txt", bftools_path="bftools"):
    command = [f"{bftools_path}/showinf", "-nopix", str(imgpath)]

    with open(output_path, "w") as outfile:
        subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT)

    print("Saved metadata to out/metadata.txt")


def save_center_crop(imgpath, series, width, height, z_depths, dstdir, bftools_path="bftools"):
    # Calculate the center crop coordinates for a 4096x4096 region
    crop_x = max(0, width // 2 - 4096 // 2)
    crop_y = max(0, height // 2 - 4096 // 2)
    crop_width = min(4096, width)
    crop_height = min(4096, height)
    
    # Loop through each Z depth and save the corresponding image
    for z in range(z_depths):
        output_path = dstdir / f"img_center_crop_z{z}.tiff"
        command = [
            f"{bftools_path}/bfconvert",
            f"-series", str(series),
            f"-crop", f"{crop_x},{crop_y},{crop_width},{crop_height}",
            f"-z", str(z),
            str(imgpath),
            str(output_path)
        ]

        subprocess.run(command)
        print(f"Saved center crop for Z depth {z} to {output_path}")


def main(imgpath):
    imgpath = Path(imgpath)
    assert imgpath.exists()
    dstdir = Path("out")
    if dstdir.exists():
        shutil.rmtree(dstdir)
    dstdir.mkdir(exist_ok=True)
    run_showinf(imgpath)  # extract metadata to file
    largest_series, largest_width, largest_height, z_depths = extract_largest_image_info()
    print(f"Largest Series: {largest_series}, Width: {largest_width}, Height: {largest_height}, Z Depths: {z_depths}")
    save_center_crop(imgpath, largest_series, largest_width, largest_height, z_depths, dstdir)  # extract center crop for all depths


if __name__ == "__main__":
    Fire(main)
