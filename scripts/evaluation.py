import os
import cv2
import torch
import numpy as np
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

# Configuration
model_dir = "/Users/njany/Documents/instasun-project/detectron2-service/models/new"  # Path to the directory where the model is saved
test_image_path = "/Users/njany/Documents/instasun-project/detectron2-service/tests/test_image.png"  # Path to the test image
output_image_path = "/Users/njany/Documents/instasun-project/detectron2-service/tests/processed_image.jpg"  # Path to save the output image

# Register the dataset if not already registered
register_coco_instances("roof_test", {}, "/Users/njany/Documents/instasun-project/detectron2-service/data_new/data/coco_val.json", "/Users/njany/Documents/instasun-project/detectron2-service/data_new/val")

# Load the configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")  # Path to the final trained model weights
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Number of classes (as per the training configuration)
cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference if GPU is not available

# Create predictor
predictor = DefaultPredictor(cfg)

# Load the test image
image = cv2.imread(test_image_path)
if image is None:
    raise FileNotFoundError(f"Test image not found at {test_image_path}")

# Perform inference
outputs = predictor(image)

# Filter out only the "ridge" class
ridge_class_id = 5  # Assuming "ridge" is class 5
ridge_indices = [i for i, class_id in enumerate(outputs["instances"].pred_classes) if class_id == ridge_class_id]
ridge_predictions = outputs["instances"][ridge_indices]

# Debugging information
print(f"Total ridges detected: {len(ridge_predictions)}")
for i in range(len(ridge_predictions)):
    print(f"Ridge {i}:")
    print(f"  Bounding box: {ridge_predictions.pred_boxes[i]}")
    print(f"  Score: {ridge_predictions.scores[i]}")
    print(f"  Mask shape: {ridge_predictions.pred_masks[i].shape}")
    print(f"  Mask data (sum of mask values): {ridge_predictions.pred_masks[i].sum()}")

# Define the mapping for class names
class_names = ["panels", "obstacles", "roof", "hip", "dormer", "ridge", "valley"]

# Set the metadata for visualization
metadata = MetadataCatalog.get("roof_test").set(thing_classes=class_names)

# Visualize the results
visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
visualizer = visualizer.draw_instance_predictions(ridge_predictions.to("cpu"))
output_image = visualizer.get_image()[:, :, ::-1]

# Save the output image
cv2.imwrite(output_image_path, output_image)
print(f"Output image saved to {output_image_path}")

# Show the output image (optional)
cv2.imshow("Output", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
