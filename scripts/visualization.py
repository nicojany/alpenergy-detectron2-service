import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

def setup_cfg():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "../models/model_final.pth"  # Adjust the path as necessary
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    return cfg

def visualize_predictions():
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    # Define metadata for the custom dataset
    custom_metadata = {
        "thing_classes": ["obstacle", "roof"]
    }
    
    img_path = "/Users/njany/Documents/instasun-project/detectron2-service/data/test_image.png"  # Replace with the path to your test image
    im = cv2.imread(img_path)
    outputs = predictor(im)
    
    # Filter out instances with a score less than 0.80
    high_confidence_instances = outputs["instances"][outputs["instances"].scores > 0.80]
    
    v = Visualizer(im[:, :, ::-1], metadata=custom_metadata, scale=1.2)
    out = v.draw_instance_predictions(high_confidence_instances.to("cpu"))
    cv2.imshow("Model Predictions", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_predictions()
