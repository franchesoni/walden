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


def main():
    # Example usage
    bboxes = [
        (54740, 36060, 50, 68),  # row, col, height, width
        (54536, 35848, 80, 91),
        (54300, 35775, 112, 1200),
    ]
    classes = [-1, -1, 0]  # Class IDs

    # Mapping class IDs to classification names (optional)
    class_map = {
        0: {"name": "Small", "color": [50, 50, 50]},
        1: {"name": "Big", "color": [100, 0, 0]},
    }

    # Example usage
    csv_file = "out/cells/global_bboxes.csv"  # Replace with your file path
    bboxes = load_bounding_boxes_csv(csv_file)
    classes = [(bbox[2] * bbox[3] > 100 * 100) * 1 for bbox in bboxes]
    print(bboxes)

    bounding_boxes_to_geojson(
        bboxes, classes, class_map=class_map, plane_info={"c": -1, "z": 7, "t": 0}
    )


main()
