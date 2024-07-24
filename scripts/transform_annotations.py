import json

def transform_annotations(input_json_path, output_json_path):
    with open(input_json_path) as f:
        data = json.load(f)
    
    for annotation in data['annotations']:
        if "segmentation" not in annotation:
            annotation["segmentation"] = []
        if "keypoints" not in annotation:
            annotation["keypoints"] = [0, 0, 0, 0, 0, 0]
        if "num_keypoints" not in annotation:
            annotation["num_keypoints"] = 0
        
        # Ensure correct dummy keypoints and num_keypoints for segmentation annotations
        if annotation["segmentation"]:
            annotation["keypoints"] = [0, 0, 0, 0, 0, 0]
            annotation["num_keypoints"] = 2
        # Ensure correct segmentation array for keypoint annotations
        elif annotation["keypoints"] and not annotation["segmentation"]:
            annotation["segmentation"] = []

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    train_input = "/Users/njany/Documents/instasun-project/detectron2-service/data/test/data/coco_train.json"
    val_input = "/Users/njany/Documents/instasun-project/detectron2-service/data/test/data/coco_val.json"
    
    train_output = "/Users/njany/Documents/instasun-project/detectron2-service/data/test/data/coco_train_transformed.json"
    val_output = "/Users/njany/Documents/instasun-project/detectron2-service/data/test/data/coco_val_transformed.json"
    
    transform_annotations(train_input, train_output)
    transform_annotations(val_input, val_output)

    print(f"Transformed training annotations saved to {train_output}")
    print(f"Transformed validation annotations saved to {val_output}")

if __name__ == "__main__":
    main()
