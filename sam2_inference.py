import tqdm
from pathlib import Path
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys
import h5py
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.amg import build_all_layer_point_grids
import csv


def sort_paths_by_dimensions(paths):
    """
    Sort a list of paths numerically based on the dimensions in the filename.

    Args:
        paths (list of Path): List of PosixPath or file paths in the format tile_<dim1>_<dim2>.jpeg.

    Returns:
        list of Path: Sorted list of paths.
    """

    def extract_dimensions(path):
        # Extract the part after "tile_" and before ".jpeg"
        filename = path.stem  # Get the file name without extension
        if filename.startswith("tile_"):
            dims = filename[len("tile_") :].split("_")
            return int(dims[0]), int(dims[1])  # Convert dimensions to integers
        raise ValueError(f"unexpected path {path}")

    # Sort paths using the extracted dimensions
    return sorted(paths, key=extract_dimensions)


def append_to_h5(new_data, feat_dim, filename="out/cells/features.h5"):
    # Create or append to an HDF5 file
    with h5py.File(filename, "a") as f:
        if "dataset" not in f:
            # Create dataset if it doesn't exist
            dset = f.create_dataset(
                "dataset",
                shape=(0, feat_dim),
                maxshape=(None, feat_dim),
                dtype="float32",
            )
        else:
            dset = f["dataset"]

        # Resize dataset to accommodate new row
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1, :] = new_data


def append_to_csv(global_bbox, filename="out/cells/global_bboxes.csv"):
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(global_bbox)


np.random.seed(3)


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def bbox_area(bbox):
    return bbox[2] * bbox[3]


def bbox_intersection(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return (x_right - x_left) * (y_bottom - y_top)


def main(sam_size="tiny", dino_size="small", vis=False, device="cuda", reset=True):
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if dino_size == "small":
        dino = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_reg", force_reload=False
        )
        dino = dino.to(device)
        dino_dim = 384
    else:
        raise ValueError(f"{dino_size} not recognized, should be in ['small']")
    dino.eval()

    if sam_size == "tiny":
        sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    elif sam_size == "small":
        sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif sam_size == "base":
        sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    else:
        raise ValueError(
            f"{sam_size} not recognized, should be in ['tiny', 'small', 'base']"
        )

    sam2_checkpoint = os.path.abspath(sam2_checkpoint)

    model = build_sam2(
        model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )

    print("generating masks...")
    point_grids = build_all_layer_point_grids(
        n_per_side=32, n_layers=0, scale_per_layer=1
    )
    point_grids = [
        p * 0.5 + 0.25 for p in point_grids
    ]  # the grid is only on a center crop
    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        point_grids=point_grids,
        points_per_side=None,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.90,
        box_nms_thresh=0.3,
        min_mask_region_area=14,  # 14x14
    )

    # filter masks mostly outside
    center_bbox = [256, 256, 512, 512]  # [x_min, y_min, width, height]

    # generate sam masks
    if reset and Path("out/cells").exists():
        import shutil

        shutil.rmtree("out/cells")
    Path("out/cells").mkdir(exist_ok=True, parents=True)
    tiles = sort_paths_by_dimensions(
        list(Path("overlapping_tiles").glob("tile_*.jpeg"))
    )
    tile_row, tile_col = 0, 0
    for tile_path in tqdm.tqdm(tiles):
        tile_row, tile_col = int(tile_path.name.split("_")[1]), int(
            tile_path.name.split("_")[2].split(".")[0]
        )
        process_tile(
            tile_row,
            tile_col,
            mask_generator,
            center_bbox,
            img_transform,
            device,
            dino,
            dino_dim,
            vis,
        )


def process_tile(
    tile_row,
    tile_col,
    mask_generator,
    center_bbox,
    img_transform,
    device,
    dino,
    dino_dim,
    vis,
):
    image1024 = Image.open(f"overlapping_tiles/tile_{tile_row}_{tile_col}.jpeg")
    masks = mask_generator.generate(np.array(image1024))

    # filter out those outside the center crop
    filtered_masks = []
    for mask in masks:
        bbox = mask["bbox"]
        intersection = bbox_intersection(bbox, center_bbox)
        if intersection >= 0.5 * bbox_area(bbox):
            filtered_masks.append(mask)

    # compute global bbox
    for mask in filtered_masks:
        x_min, y_min, width, height = mask["bbox"]
        mask["global_bbox"] = [
            tile_row + y_min,
            tile_col + x_min,
            height,
            width,
        ]  # [row, col, height, width]

    # forward dino over center crop of size 644
    image644 = image1024.crop((190, 190, 834, 834))
    with torch.no_grad():
        input_img = img_transform(image644).reshape(1, 3, 644, 644).to(device)
        feats = dino.forward_features(input_img)["x_norm_patchtokens"].reshape(
            1, 46, 46, dino_dim
        )

    # compute dino avg feature for each mask
    features = feats.squeeze(0).cpu().numpy()  # Shape: [46, 46, dino_dim]
    for mask in filtered_masks:
        segmentation = mask["segmentation"]  # Shape: [1024, 1024]

        # Crop the segmentation to the center crop coordinates
        seg_crop = segmentation[190:834, 190:834]  # Shape: [644, 644]

        # Downsample the segmentation to match the DINO feature map size
        seg_downsampled = cv2.resize(
            (seg_crop * 255).astype(np.uint8), (46, 46), interpolation=cv2.INTER_LINEAR
        )
        seg_downsampled_bool = seg_downsampled > 127  # Shape: [46, 46]

        # Compute the average feature within the mask
        masked_features = features[seg_downsampled_bool]
        if masked_features.size == 0:
            print(
                f"mask at {mask['global_bbox']} has zero downsampled area, skipping..."
            )
            continue
        avg_feature = masked_features.mean(axis=0)

        append_to_h5(avg_feature, dino_dim)
        append_to_csv(mask["global_bbox"])

    if vis:
        plt.figure(figsize=(20, 20))
        plt.imshow(image1024)
        show_anns(filtered_masks)
        plt.axis("off")
        plt.savefig("out.png")
        plt.close()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
