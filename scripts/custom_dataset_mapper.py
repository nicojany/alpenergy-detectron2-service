import detectron2.data.detection_utils as utils
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
import numpy as np
import torch

class CustomDatasetMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.augmentations = T.AugmentationList(utils.build_augmentation(cfg, is_train))
        self.img_format = cfg.INPUT.FORMAT
        self.keypoint_hflip_indices = (
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).keypoint_flip_map
            if cfg.MODEL.KEYPOINT_ON and len(cfg.DATASETS.TRAIN)
            else None
        )

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = utils.annotations_to_instances(annos, image_shape)

        # Ensure all required fields are present
        if not hasattr(instances, "gt_masks"):
            instances.gt_masks = torch.zeros((len(instances), *image_shape), dtype=torch.uint8)

        dataset_dict["instances"] = instances
        return dataset_dict
