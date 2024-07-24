import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for the datasets
base_folder = '/Users/njany/Documents/instasun-project/detectron2-service/data_new/306022_Roof Detection'
train_ann_folder = os.path.join(base_folder, 'Train/ann')
train_img_folder = os.path.join(base_folder, 'Train/img')
val_ann_folder = os.path.join(base_folder, 'Validation/ann')
val_img_folder = os.path.join(base_folder, 'Validation/img')
train_output_file = os.path.join(base_folder, 'coco_train.json')
val_output_file = os.path.join(base_folder, 'coco_val.json')

# Debugging prints to verify paths
logging.debug(f"Train Annotation folder: {train_ann_folder}")
logging.debug(f"Train Image folder: {train_img_folder}")
logging.debug(f"Train Output file: {train_output_file}")
logging.debug(f"Validation Annotation folder: {val_ann_folder}")
logging.debug(f"Validation Image folder: {val_img_folder}")
logging.debug(f"Validation Output file: {val_output_file}")

# Check if the paths exist
if not os.path.exists(train_ann_folder):
    logging.error(f"Train Annotation folder does not exist: {train_ann_folder}")
if not os.path.exists(train_img_folder):
    logging.error(f"Train Image folder does not exist: {train_img_folder}")
if not os.path.exists(val_ann_folder):
    logging.error(f"Validation Annotation folder does not exist: {val_ann_folder}")
if not os.path.exists(val_img_folder):
    logging.error(f"Validation Image folder does not exist: {val_img_folder}")
if not os.path.exists(os.path.dirname(train_output_file)):
    logging.error(f"Train Output directory does not exist: {os.path.dirname(train_output_file)}")
if not os.path.exists(os.path.dirname(val_output_file)):
    logging.error(f"Validation Output directory does not exist: {os.path.dirname(val_output_file)}")

# Initialize COCO JSON structure
coco_data_template = {
    "info": {
        "description": "Converted from Supervisely",
        "url": "",
        "version": "1.0",
        "year": 2024,
        "contributor": "Supervisely",
        "date_created": ""
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"supercategory": "none", "id": 1, "name": "panels"},
        {"supercategory": "none", "id": 2, "name": "obstacles"},
        {"supercategory": "none", "id": 3, "name": "roof"},
        {"supercategory": "none", "id": 4, "name": "hip"},
        {"supercategory": "none", "id": 5, "name": "dormer"},
        {"supercategory": "none", "id": 6, "name": "ridge"},
        {"supercategory": "none", "id": 7, "name": "valley"}
    ]
}

# Create a map for class titles to category IDs
category_map = {
    "panels": 1,
    "obstacles": 2,
    "roof": 3,
    "hip": 4,
    "dormer": 5,
    "ridge": 6,
    "valley": 7
}

# Helper function to create a small buffer around a line
def create_buffered_polygon(points, buffer_size=5):
    polygon = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / length, dy / length  # unit vector
        polygon.append((x1 - uy * buffer_size, y1 + ux * buffer_size))
        polygon.append((x1 + uy * buffer_size, y1 - ux * buffer_size))
    x1, y1 = points[-1]
    polygon.append((x1 - uy * buffer_size, y1 + ux * buffer_size))
    polygon.append((x1 + uy * buffer_size, y1 - ux * buffer_size))
    return polygon

def convert_to_coco(ann_folder, img_folder, output_file, coco_data):
    # Initialize counters
    annotation_id = 1
    image_id = 1

    # Process each JSON file in the annotation folder
    for filename in os.listdir(ann_folder):
        if filename.endswith('.json'):
            logging.debug(f"Processing file: {filename}")
            with open(os.path.join(ann_folder, filename)) as f:
                supervisely_data = json.load(f)

            # Extract the image filename and remove any double .png suffix
            image_filename = filename.replace('.json', '.png')  # Assuming images are in PNG format
            if image_filename.endswith('.png.png'):
                image_filename = image_filename.replace('.png.png', '.png')

            # Add the image information to COCO structure
            coco_image = {
                "license": 0,
                "file_name": image_filename,  # Use the corrected image filename
                "coco_url": "",
                "height": supervisely_data['size']['height'],
                "width": supervisely_data['size']['width'],
                "date_captured": "",
                "flickr_url": "",
                "id": image_id
            }
            coco_data['images'].append(coco_image)

            # Convert annotations
            for obj in supervisely_data['objects']:
                category_id = category_map[obj['classTitle']]
                if obj['geometryType'] == 'polygon':
                    segmentation = [coord for point in obj['points']['exterior'] for coord in point]
                    bbox = [
                        min([p[0] for p in obj['points']['exterior']]),
                        min([p[1] for p in obj['points']['exterior']]),
                        max([p[0] for p in obj['points']['exterior']]) - min([p[0] for p in obj['points']['exterior']]),
                        max([p[1] for p in obj['points']['exterior']]) - min([p[1] for p in obj['points']['exterior']])
                    ]
                    area = bbox[2] * bbox[3]
                    if area == 0 or len(segmentation) < 6:  # Check for invalid segmentation
                        logging.warning(f"Filtered out annotation {annotation_id} in file {filename} due to invalid segmentation.")
                        continue
                    coco_annotation = {
                        "segmentation": [segmentation],
                        "area": area,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": bbox,
                        "category_id": category_id,
                        "id": annotation_id
                    }
                elif obj['geometryType'] == 'line':
                    points = obj['points']['exterior']
                    if len(points) < 2:  # A line needs at least 2 points
                        logging.warning(f"Filtered out annotation {annotation_id} in file {filename} due to insufficient points in line.")
                        continue
                    polygon = create_buffered_polygon(points)
                    segmentation = [coord for point in polygon for coord in point]
                    bbox = [
                        min([p[0] for p in polygon]),
                        min([p[1] for p in polygon]),
                        max([p[0] for p in polygon]) - min([p[0] for p in polygon]),
                        max([p[1] for p in polygon]) - min([p[1] for p in polygon])
                    ]
                    area = bbox[2] * bbox[3]
                    if area == 0:
                        logging.warning(f"Filtered out annotation {annotation_id} in file {filename} due to zero area.")
                        continue
                    coco_annotation = {
                        "segmentation": [segmentation],
                        "area": area,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": bbox,
                        "category_id": category_id,
                        "id": annotation_id
                    }
                coco_data['annotations'].append(coco_annotation)
                annotation_id += 1
            
            image_id += 1

    # Save COCO JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)

    logging.info(f"COCO annotations saved to: {output_file}")

# Initialize COCO structures for train and val
coco_train_data = json.loads(json.dumps(coco_data_template))  # Deep copy
coco_val_data = json.loads(json.dumps(coco_data_template))  # Deep copy

# Convert train and val datasets
convert_to_coco(train_ann_folder, train_img_folder, train_output_file, coco_train_data)
convert_to_coco(val_ann_folder, val_img_folder, val_output_file, coco_val_data)
