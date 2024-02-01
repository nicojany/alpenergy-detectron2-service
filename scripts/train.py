from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor

def register_dataset(train_json, train_dir, val_json, val_dir):
    register_coco_instances("my_dataset_train", {}, train_json, train_dir)
    register_coco_instances("my_dataset_val", {}, val_json, val_dir)

def train_model(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def evaluate_model(cfg):
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    inference_on_dataset(DefaultPredictor(cfg).model, val_loader, evaluator)

def main(train_json, val_json, train_dir, val_dir, train=True, evaluate=True):
    register_dataset(train_json, train_dir, val_json, val_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cpu"

    if train:
        train_model(cfg)

    if evaluate:
        evaluate_model(cfg)

if __name__ == '__main__':
    main("../data/annotations/instances_train.json",  
         "../data/annotations/instances_val.json",    
         "../data/train",                            
         "../data/val",                            
         train=True, 
         evaluate=True)
