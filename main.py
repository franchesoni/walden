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

    return largest_series, z_depths


def run_showinf(imgpath, output_path="out/metadata.txt", bftools_path="bftools"):
    command = [f"{bftools_path}/showinf", "-nopix", str(imgpath)]

    with open(output_path, "w") as outfile:
        subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT)

    print("Saved metadata to out/metadata.txt")


def main(imgpath):
    imgpath = Path(imgpath)
    assert imgpath.exists()
    dstdir = Path("out")
    if dstdir.exists():
        shutil.rmtree(dstdir)
    dstdir.mkdir(exist_ok=True)
    run_showinf(imgpath)  # extract metadata to file
    largest_series, z_depths = extract_largest_image_info()
    print(f"Largest Series: {largest_series}, Z Depths: {z_depths}")


if __name__ == "__main__":
    Fire(main)
