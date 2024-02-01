import json

with open("instances_val.json", "r") as f:
    data = json.load(f)

for ann in data["annotations"]:
    if "bbox" in ann:
        x, y, w, h = ann["bbox"]
        ann["segmentation"] = [[[x, y, x+w, y, x+w, y+h, x, y+h]]]  # Convert bbox to polygon

with open("instances_val_polygon.json", "w") as f:
    json.dump(data, f)
