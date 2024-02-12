import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

def load_predictor(model_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.DATASETS.TRAIN = ("roof_train",)  # Update this if necessary
    MetadataCatalog.get("roof_train").set(thing_classes=["obstacle", "roof"])
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def process_image(cfg, predictor, input_image_path, output_image_path):
    image = cv2.imread(input_image_path)
    outputs = predictor(image)
    
    # Visualization
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = v.get_image()[:, :, ::-1]
    
    # Save the processed image
    cv2.imwrite(output_image_path, result_image)

if __name__ == "__main__":
    # Update with your model path and input/output image paths
    model_path = "../models/model_final.pth"
    input_image_path = "../data/test_image.png"
    output_image_path = "../data/processed_image.png"
    
    cfg, predictor = load_predictor(model_path)
    process_image(cfg, predictor, input_image_path, output_image_path)
