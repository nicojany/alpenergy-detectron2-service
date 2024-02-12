import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import numpy as np
import cv2
import random

# Import Detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def setup_cfg():
    # Load the configuration from file
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Set the dataset paths
    cfg.DATASETS.TRAIN = ("roof_train",)
    cfg.DATASETS.TEST = ("roof_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # Set the model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Set the number of classes (2 for roof and obstacles)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.DEVICE = "cpu"

    # Set the output directory
    cfg.OUTPUT_DIR = "/Users/njany/Documents/instasun-project/detectron2-service/models"

    return cfg

def main():
    # Register the datasets
    register_coco_instances("roof_train", {}, "/Users/njany/Documents/instasun-project/detectron2-service/data/annotations/instances_train.json", "/Users/njany/Documents/instasun-project/detectron2-service/data/train")
    register_coco_instances("roof_val", {}, "/Users/njany/Documents/instasun-project/detectron2-service/data/annotations/instances_val.json", "/Users/njany/Documents/instasun-project/detectron2-service/data/val")

    cfg = setup_cfg()

    # Create a trainer and start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
