import shutil
from pathlib import Path
import subprocess
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from fire import Fire
import re
import cv2
import numpy as np
from PIL import Image, ImageFile
import warnings

# Increase the decompression bomb limit to handle large images
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def extract_largest_image_info(metadata_path):
    metadata_path = str(metadata_path)
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


def run_showinf(imgpath, output_path, bftools_path="bftools"):
    command = [f"{bftools_path}/showinf", "-nopix", str(imgpath)]

    with open(output_path, "w") as outfile:
        subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT)

    print("Saved metadata to metadata.txt")



def save_center_crop_parallel(
    imgpath, series, width, height, z_depths, dstdir, bftools_path="bftools", max_workers=4
):
    """
    Saves center-cropped images for each Z depth in parallel.

    Parameters:
    - imgpath (str or Path): Path to the source image.
    - series (int): Series number for bfconvert.
    - width (int): Width of the source image.
    - height (int): Height of the source image.
    - z_depths (int): Number of Z depths to process.
    - dstdir (str or Path): Destination directory to save cropped images.
    - bftools_path (str): Path to the bfconvert tool.
    - max_workers (int): Maximum number of threads to use for parallel processing.
    """
    
    # Calculate the center crop coordinates for a 4096x4096 region
    crop_x = max(0, width // 2 - 4096 // 2)
    crop_y = max(0, height // 2 - 4096 // 2)
    crop_width = min(4096, width)
    crop_height = min(4096, height)

    # Ensure the destination directory exists
    dstdir = Path(dstdir)
    dstdir.mkdir(parents=True, exist_ok=True)

    def process_z(z):
        """
        Processes a single Z depth by running the bfconvert command.

        Parameters:
        - z (int): The Z depth to process.

        Returns:
        - tuple: (z, output_path) if successful.
        """
        output_path = dstdir / f"img_center_crop_z{z}.tiff"
        command = [
            f"{bftools_path}/bfconvert",
            "-series", str(series),
            "-crop", f"{crop_x},{crop_y},{crop_width},{crop_height}",
            "-z", str(z),
            str(imgpath),
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
            return z, output_path
        except subprocess.CalledProcessError as e:
            print(f"Error processing Z depth {z}: {e}")
            return z, None

    # Use ThreadPoolExecutor to parallelize the processing of Z depths
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_z = {executor.submit(process_z, z): z for z in range(z_depths)}
        
        # As each task completes, handle the result
        for future in as_completed(future_to_z):
            z = future_to_z[future]
            try:
                z_processed, output_path = future.result()
                if output_path:
                    print(f"Saved center crop for Z depth {z_processed} to {output_path}")
                else:
                    print(f"Failed to save center crop for Z depth {z_processed}")
            except Exception as exc:
                print(f"Z depth {z} generated an exception: {exc}")



def save_center_crop(
    imgpath, series, width, height, z_depths, dstdir, bftools_path="bftools", max_workers=None
):
    if max_workers is not None:
        return save_center_crop_parallel(imgpath, series, width, height, z_depths, dstdir, bftools_path="bftools", max_workers=max_workers)
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
            f"-series",
            str(series),
            f"-crop",
            f"{crop_x},{crop_y},{crop_width},{crop_height}",
            f"-z",
            str(z),
            str(imgpath),
            str(output_path),
        ]

        subprocess.run(command)
        print(f"Saved center crop for Z depth {z} to {output_path}")


def save_full_image_crops(
    imgpath, series, width, height, best_focuses, dstdir, bftools_path="bftools"
):
    # Define maximum tile size
    max_tile_size = 2**14

    # Loop through each best focus and divide the image into tiles
    for z in best_focuses:
        # Calculate number of tiles in x and y direction
        num_tiles_x = (width + max_tile_size - 1) // max_tile_size
        num_tiles_y = (height + max_tile_size - 1) // max_tile_size

        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Calculate crop coordinates
                crop_x = i * max_tile_size
                crop_y = j * max_tile_size
                crop_width = min(max_tile_size, width - crop_x)
                crop_height = min(max_tile_size, height - crop_y)

                # Set output paths
                tiff_output_path = dstdir / f"img_full_crop_z{z}_tile_{i}_{j}.tiff"
                jpeg_output_path = dstdir / f"img_full_crop_z{z}_tile_{i}_{j}.jpeg"

                # Run bfconvert to save TIFF tile
                command = [
                    f"{bftools_path}/bfconvert",
                    f"-series",
                    str(series),
                    f"-crop",
                    f"{crop_x},{crop_y},{crop_width},{crop_height}",
                    f"-z",
                    str(z),
                    str(imgpath),
                    str(tiff_output_path),
                ]

                subprocess.run(command)
                print(
                    f"Saved TIFF tile for Z depth {z}, tile ({i}, {j}) to {tiff_output_path}"
                )

                # Convert TIFF to JPEG using PIL and then remove the TIFF
                try:
                    with Image.open(tiff_output_path) as img:
                        img.convert("RGB").save(jpeg_output_path, "JPEG")
                    print(
                        f"Converted TIFF to JPEG for Z depth {z}, tile ({i}, {j}) to {jpeg_output_path}"
                    )
                except Image.DecompressionBombWarning as e:
                    print(f"Warning: {e} for image {tiff_output_path}")

                # Remove the TIFF file
                tiff_output_path.unlink()
                print(f"Removed TIFF file: {tiff_output_path}")


# Move the process_tile function to the top level so it can be pickled
def process_tile(args):
    z, i, j, imgpath, series, width, height, dstdir_str, bftools_path, max_tile_size = (
        args
    )

    # Convert dstdir back to Path object
    dstdir = Path(dstdir_str)

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Calculate crop coordinates
    crop_x = i * max_tile_size
    crop_y = j * max_tile_size
    crop_width = min(max_tile_size, width - crop_x)
    crop_height = min(max_tile_size, height - crop_y)

    # Set output paths
    tiff_output_path = dstdir / f"img_full_crop_z{z}_tile_{i}_{j}.tiff"
    jpeg_output_path = dstdir / f"img_full_crop_z{z}_tile_{i}_{j}.jpeg"

    # Run bfconvert to save TIFF tile
    command = [
        f"{bftools_path}/bfconvert",
        "-series",
        str(series),
        "-crop",
        f"{crop_x},{crop_y},{crop_width},{crop_height}",
        "-z",
        str(z),
        str(imgpath),
        str(tiff_output_path),
    ]

    subprocess.run(command, check=True)
    print(f"Saved TIFF tile for Z depth {z}, tile ({i}, {j}) to {tiff_output_path}")

    # Convert TIFF to JPEG using PIL and then remove the TIFF
    try:
        with Image.open(tiff_output_path) as img:
            img.convert("RGB").save(jpeg_output_path, "JPEG")
        print(
            f"Converted TIFF to JPEG for Z depth {z}, tile ({i}, {j}) to {jpeg_output_path}"
        )
    except Image.DecompressionBombWarning as e:
        print(f"Warning: {e} for image {tiff_output_path}")

    # Remove the TIFF file
    tiff_output_path.unlink()
    print(f"Removed TIFF file: {tiff_output_path}")


def save_full_image_crops_parallel(
    imgpath, series, width, height, best_focuses, dstdir, bftools_path="bftools"
):
    # Define maximum tile size
    max_tile_size = 2**14

    # Calculate number of tiles in x and y direction
    num_tiles_x = (width + max_tile_size - 1) // max_tile_size
    num_tiles_y = (height + max_tile_size - 1) // max_tile_size

    # Prepare list of tasks
    tasks = []

    for z in best_focuses:
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                tasks.append(
                    (
                        z,
                        i,
                        j,
                        str(imgpath),
                        series,
                        width,
                        height,
                        str(dstdir),
                        bftools_path,
                        max_tile_size,
                    )
                )

    # Use ProcessPoolExecutor to parallelize
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_tile, tasks))


def compute_sharpness(image_path):
    # Load the image in grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Use the Laplacian method to calculate sharpness
    return cv2.Laplacian(image, cv2.CV_64F).var()


def select_best_focus_images(dstdir, z_depths, n=1):
    sharpness_values = []
    # Calculate sharpness for each image
    for z in range(z_depths):
        image_path = dstdir / f"img_center_crop_z{z}.tiff"
        sharpness = compute_sharpness(image_path)
        sharpness_values.append((z, sharpness))
        print(f"Z depth {z}: Sharpness = {sharpness}")

    # Find the n consecutive images with the highest sharpness sum
    best_start_index = 0
    max_sharpness_sum = sum([sharpness_values[i][1] for i in range(n)])
    for i in range(1, len(sharpness_values) - n + 1):
        sharpness_sum = sum([sharpness_values[i + j][1] for j in range(n)])
        if sharpness_sum > max_sharpness_sum:
            max_sharpness_sum = sharpness_sum
            best_start_index = i

    best_indices = [sharpness_values[best_start_index + i][0] for i in range(n)]
    print(
        f"Best consecutive Z depths: {best_indices} with sharpness sum {max_sharpness_sum}"
    )
    return best_indices


def create_composite_image(best_focuses, width, height, dstdir, downscale_factor=4):
    # Create a blank canvas for the composite image
    composite_width = width // downscale_factor
    composite_height = height // downscale_factor
    composite_image = np.zeros((composite_height, composite_width, 3), dtype=np.float32)
    count_matrix = np.zeros((composite_height, composite_width), dtype=np.float32)

    # Define maximum tile size
    max_tile_size = 2**14

    # Loop through each best focus and average the downscaled tiles into the composite image
    for z in best_focuses:
        num_tiles_x = (width + max_tile_size - 1) // max_tile_size
        num_tiles_y = (height + max_tile_size - 1) // max_tile_size

        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                jpeg_output_path = dstdir / f"img_full_crop_z{z}_tile_{i}_{j}.jpeg"
                print(f"Processing img_full_crop_z{z}_tile_{i}_{j}.jpeg", end="\r")
                if not jpeg_output_path.exists():
                    continue

                # Open the JPEG tile and downscale it
                with Image.open(jpeg_output_path) as tile:
                    tile_downscaled = tile.resize(
                        (
                            tile.width // downscale_factor,
                            tile.height // downscale_factor,
                        ),
                        Image.LANCZOS,
                    )
                    tile_array = np.array(tile_downscaled, dtype=np.float32)

                    # Calculate where to paste the downscaled tile in the composite image
                    paste_x = (i * max_tile_size) // downscale_factor
                    paste_y = (j * max_tile_size) // downscale_factor

                    # Update the composite image by averaging values
                    composite_image[
                        paste_y : paste_y + tile_array.shape[0],
                        paste_x : paste_x + tile_array.shape[1],
                        :,
                    ] += tile_array
                    count_matrix[
                        paste_y : paste_y + tile_array.shape[0],
                        paste_x : paste_x + tile_array.shape[1],
                    ] += 1

    # Avoid division by zero and normalize the composite image
    count_matrix[count_matrix == 0] = 1
    composite_image /= count_matrix[:, :, np.newaxis]

    # Convert the composite image to uint8 and save it
    composite_image = np.clip(composite_image, 0, 255).astype(np.uint8)
    composite_output_path = dstdir / "composite_image.jpeg"
    Image.fromarray(composite_image).save(composite_output_path, "JPEG")
    print(f"Saved composite image to {composite_output_path}")


def vsi_to_jpegs(imgpath, dstname, n_focuses=1, reset=False, parallel=False):
    imgpath = Path(imgpath)
    assert imgpath.exists()
    dstdir = Path("out") / dstname
    if reset and dstdir.exists():
        shutil.rmtree(dstdir)
    dstdir.mkdir(exist_ok=True)
    run_showinf(imgpath, str(dstdir / "metadata.txt"))  # extract metadata to file
    largest_series, largest_width, largest_height, z_depths = (
        extract_largest_image_info(dstdir / "metadata.txt")
    )
    print(
        f"Largest Series: {largest_series}, Width: {largest_width}, Height: {largest_height}, Z Depths: {z_depths}"
    )
    if reset or not (dstdir / "img_center_crop_z0.tiff").exists():
        save_center_crop(
            imgpath, largest_series, largest_width, largest_height, z_depths, dstdir, max_workers=24 if parallel else None
        )  # extract center crop for all depths
    best_focuses = select_best_focus_images(
        dstdir, z_depths, n_focuses
    )  # find the best focus images
    if reset or not len(list(dstdir.glob("img_full_crop_z*_tile_0_0.jpeg"))):
        if parallel:
            save_full_image_crops_parallel(
                imgpath,
                largest_series,
                largest_width,
                largest_height,
                best_focuses,
                dstdir,
            )
        else:
            save_full_image_crops(
                imgpath,
                largest_series,
                largest_width,
                largest_height,
                best_focuses,
                dstdir,
            )  # save full image crops for best focuses
    if reset or not (dstdir / "composite_image.jpeg").exists():
        create_composite_image(
            best_focuses, largest_width, largest_height, dstdir, downscale_factor=32
        )  # create a composite lower resolution image
    # save metadata
    metadata_output_path = dstdir / "extracted_metadata.txt"
    with open(metadata_output_path, "w") as metadata_file:
        metadata_file.write(
            f"Largest Series: {largest_series}\nWidth: {largest_width}\nHeight: {largest_height}\nZ Depths: {z_depths}\nBest focuses: {best_focuses}"
        )
    print(f"Saved extracted metadata to {metadata_output_path}")


if __name__ == "__main__":
    Fire(vsi_to_jpegs)
