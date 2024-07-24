import json

# Load Supervisely JSON file
with open('supervisely_export.json') as f:
    supervisely_data = json.load(f)

# Initialize COCO JSON structure
coco_data = {
    "info": {
        "description": "Converted from Supervisely",
        "url": "",
        "version": "1.0",
        "year": 2024,
        "contributor": "Supervisely",
        "date_created": supervisely_data['createdAt']
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {"supercategory": "none", "id": 1, "name": "hip"},
        {"supercategory": "none", "id": 2, "name": "obstacle"},
        {"supercategory": "none", "id": 3, "name": "roof"},
        {"supercategory": "none", "id": 4, "name": "dormer"},
        {"supercategory": "none", "id": 5, "name": "ridge"},
        {"supercategory": "none", "id": 6, "name": "valley"}
    ]
}

# Create a map for class titles to category IDs
category_map = {
    "hip": 1,
    "obstacle": 2,
    "roof": 3,
    "dormer": 4,
    "ridge": 5,
    "valley": 6
}

# Add the image information to COCO structure
coco_image = {
    "license": 0,
    "file_name": supervisely_data['imageName'],
    "coco_url": "",
    "height": supervisely_data['annotation']['size']['height'],
    "width": supervisely_data['annotation']['size']['width'],
    "date_captured": supervisely_data['createdAt'],
    "flickr_url": "",
    "id": supervisely_data['imageId']
}
coco_data['images'].append(coco_image)

# Convert annotations
annotation_id = 1
for obj in supervisely_data['annotation']['objects']:
    category_id = category_map[obj['classTitle']]
    if obj['geometryType'] == 'polygon':
        segmentation = [coord for point in obj['points']['exterior'] for coord in point]
        bbox = [
            min([p[0] for p in obj['points']['exterior']]),
            min([p[1] for p in obj['points']['exterior']]),
            max([p[0] for p in obj['points']['exterior']]) - min([p[0] for p in obj['points']['exterior']]),
            max([p[1] for p in obj['points']['exterior']]) - min([p[1] for p in obj['points']['exterior']])
        ]
        coco_annotation = {
            "segmentation": [segmentation],
            "area": 0,
            "iscrowd": 0,
            "image_id": supervisely_data['imageId'],
            "bbox": bbox,
            "category_id": category_id,
            "id": annotation_id
        }
    elif obj['geometryType'] == 'line':
        # Treat polylines as open polygons (COCO does not support polylines directly)
        segmentation = [coord for point in obj['points']['exterior'] for coord in point]
        bbox = [
            min([p[0] for p in obj['points']['exterior']]),
            min([p[1] for p in obj['points']['exterior']]),
            max([p[0] for p in obj['points']['exterior']]) - min([p[0] for p in obj['points']['exterior']]),
            max([p[1] for p in obj['points']['exterior']]) - min([p[1] for p in obj['points']['exterior']])
        ]
        coco_annotation = {
            "segmentation": [segmentation],
            "area": 0,
            "iscrowd": 0,
            "image_id": supervisely_data['imageId'],
            "bbox": bbox,
            "category_id": category_id,
            "id": annotation_id
        }
    coco_data['annotations'].append(coco_annotation)
    annotation_id += 1

# Save COCO JSON file
with open('coco_annotations.json', 'w') as f:
    json.dump(coco_data, f)
