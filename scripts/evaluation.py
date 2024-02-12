import os
import random
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.data.datasets import load_coco_json

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Same as in execute.py
    cfg.MODEL.WEIGHTS = os.path.join("/Users/njany/Documents/instasun-project/detectron2-service/models", "model_final.pth")
    cfg.MODEL.DEVICE = 'cpu'  # Use CPU
    cfg.DATASETS.TEST = ("roof_val", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # You might adjust this if needed
    return cfg

def register_dataset():
    data_dir = "/Users/njany/Documents/instasun-project/detectron2-service/data"
    json_file = os.path.join(data_dir, "annotations/instances_val.json")
    image_root = os.path.join(data_dir, "val")
    DatasetCatalog.register("roof_val", lambda: load_coco_json(json_file, image_root))
    MetadataCatalog.get("roof_val").set(thing_classes=["obstacle", "roof"])  # Same as in execute.py

def visualize_predictions(cfg, num_samples=3):
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get("roof_val")
    for d in random.sample(dataset_dicts, num_samples):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("roof_val"), 
                       scale=0.5, 
                       instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('Prediction', v.get_image()[:, :, ::-1])
        cv2.waitKey(0)

def main():
    setup_logger()
    cfg = setup_cfg()
    register_dataset()

    evaluator = COCOEvaluator("roof_val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "evaluation"))
    val_loader = build_detection_test_loader(cfg, "roof_val")
    inference_on_dataset(DefaultPredictor(cfg).model, val_loader, evaluator)

    visualize_predictions(cfg)

if __name__ == "__main__":
    main()
