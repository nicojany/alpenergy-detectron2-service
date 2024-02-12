import os
import logging
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
import cv2
from detectron2.structures import BoxMode

# Configure logging
logging.basicConfig(filename='train.log', level=logging.DEBUG)

def load_dataset(dataset_dir):
    # Get the absolute path of the directory containing the script
    script_dir = os.path.dirname(__file__)

    # Register your dataset here
    DatasetCatalog.register("my_dataset_train", lambda: load_data("train", dataset_dir))
    DatasetCatalog.register("my_dataset_val", lambda: load_data("val", dataset_dir))
    MetadataCatalog.get("my_dataset_train").set(thing_classes=["background", "obstacles", "roof"])
    MetadataCatalog.get("my_dataset_val").set(thing_classes=["background", "obstacles", "roof"])

def load_data(dataset_type, dataset_dir):
    dataset_dicts = []

    if dataset_type == "train":
        json_file = os.path.join(dataset_dir, "annotations", "instances_train.json")
        img_dir = os.path.join(dataset_dir, "train")
    elif dataset_type == "val":
        json_file = os.path.join(dataset_dir, "annotations", "instances_val.json")
        img_dir = os.path.join(dataset_dir, "val")

    with open(json_file) as f:
        data = json.load(f)

    for item in data:
        record = {}

        filename = os.path.join(img_dir, item["file_name"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = item["id"]
        record["height"] = height
        record["width"] = width

        annos = item["annotations"]
        objs = []
        for anno in annos:
            if "bbox" in anno:
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": anno["segmentation"],
                    "category_id": anno["category_id"],
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    # Get the absolute path of the directory containing the script
    script_dir = os.path.dirname(__file__)

    # Update the dataset directory with the correct path
    dataset_dir = os.path.join(script_dir, "/Users/njany/Documents/instasun-project/detectron2-service")  # Update with the correct path to your dataset directory
    cfg = get_cfg()
    cfg.merge_from_file("../configs/roof_detection.yaml")  # Path to your configuration file
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.DEVICE = "cpu"

    load_dataset(dataset_dir)  # Pass the updated dataset directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
