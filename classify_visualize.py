from pathlib import Path
import numpy as np
import h5py
import csv
import json
import uuid


def bounding_boxes_to_geojson(
    bboxes,
    classes,
    class_map=None,
    plane_info={"c": -1, "z": 0, "t": 0},
    output_file="out/cells/predictions.geojson",
):
    """
    Convert bounding boxes and classes to a list of individual GeoJSON features.

    Args:
        bboxes (list): List of bounding boxes in the format (row, col, height, width).
        classes (list): List of integer classes corresponding to the bounding boxes.
        class_map (dict, optional): Dictionary mapping class integers to classification names.
                                    Example: {0: {"name": "Background", "color": [50, 50, 50]}}.
        plane_info (dict, optional): Dictionary with additional plane info (e.g., {"c": -1, "z": 7, "t": 0}).
                                     Default is {"c": -1, "z": 0, "t": 0}.
        output_file (str): Name of the output GeoJSON file.

    Returns:
        None. Writes a list of GeoJSON features to the specified file.
    """
    features = []

    for bbox, cls in zip(bboxes, classes):
        row, col, height, width = bbox
        # Calculate polygon coordinates from bounding box
        coordinates = [
            [
                [col, row],  # Top-left
                [col + width, row],  # Top-right
                [col + width, row + height],  # Bottom-right
                [col, row + height],  # Bottom-left
                [col, row],  # Close the loop
            ]
        ]

        # Create individual feature
        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),  # Generate a unique ID for each feature
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates,
                "plane": plane_info,
            },
            "properties": {"objectType": "annotation"},
        }

        # Add classification if a class_map is provided
        if class_map and cls in class_map:
            feature["properties"]["classification"] = class_map[cls]

        # Add each feature to the list
        features.append(feature)

    # Write the list of features directly to the file
    with open(output_file, "w") as f:
        json.dump(features, f, indent=2)

    print(f"GeoJSON features saved to {output_file}")


def load_bounding_boxes_csv(csv_file):
    bboxes = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            bboxes.append([int(float(x)) for x in row])  # Convert strings to floats
    return bboxes


def main(n_clusters=10):
    # Example usage with feature clustering
    # Mapping class IDs to classification names (optional)
    colors = [color.tolist() for color in np.random.choice(range(256), size=(n_clusters, 3), replace=False)]
    class_map = {i : {"name": f"Cluster {i}", "color": colors[i]} for i in range(n_clusters)}

    csv_file = "out/cells/global_bboxes.csv"  
    bboxes = load_bounding_boxes_csv(csv_file)

    with h5py.File("out/cells/features.h5", 'r') as f:
        data = f['dataset'][:]

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=0)
    classes = kmeans.fit_predict(data).tolist()
    length = min(len(bboxes), len(classes))
    bboxes, classes = bboxes[:length], classes[:length]

    chosen_z = int(list(Path('out').glob("img_full_crop_z*_tile_0_0.jpeg"))[0].stem.split("z")[1].split("_")[0]) + 1
    bounding_boxes_to_geojson(
        bboxes, classes, class_map=class_map, plane_info={"c": -1, "z": chosen_z, "t": 0}
    )


if __name__ == '__main__':
    main()
