import json

def validate_annotations(coco_json_path):
    with open(coco_json_path) as f:
        data = json.load(f)
    
    valid = True
    image_id_to_file_name = {image['id']: image['file_name'] for image in data['images']}
    
    for annotation in data['annotations']:
        if 'keypoints' not in annotation or 'segmentation' not in annotation:
            image_id = annotation['image_id']
            print(f"Missing keypoints or segmentation in annotation ID: {annotation['id']} for image ID: {image_id} (file: {image_id_to_file_name[image_id]})")
            valid = False

        # Validate keypoint annotations
        if annotation['keypoints'] != [0, 0, 0, 0, 0, 0] and annotation['segmentation'] == []:
            if annotation['num_keypoints'] != 2:
                image_id = annotation['image_id']
                print(f"Empty segmentation but non-dummy keypoints in annotation ID: {annotation['id']} for image ID: {image_id} (file: {image_id_to_file_name[image_id]})")
                valid = False
            elif len(annotation['keypoints']) != 6:
                image_id = annotation['image_id']
                print(f"Keypoint count mismatch in annotation ID: {annotation['id']} for image ID: {image_id} (file: {image_id_to_file_name[image_id]})")
                valid = False
        
        # Validate segmentation annotations
        if annotation['segmentation']:
            if annotation['keypoints'] != [0, 0, 0, 0, 0, 0]:
                image_id = annotation['image_id']
                print(f"Non-dummy keypoints for segmentation annotation ID: {annotation['id']} for image ID: {image_id} (file: {image_id_to_file_name[image_id]})")
                valid = False
            if annotation['num_keypoints'] != 2:
                image_id = annotation['image_id']
                print(f"Segmentation annotation has incorrect num_keypoints in annotation ID: {annotation['id']} for image ID: {image_id} (file: {image_id_to_file_name[image_id]})")
                valid = False

    if valid:
        print("Validation complete: All annotations are correctly formatted.")
    else:
        print("Validation complete: Some annotations are incorrectly formatted.")

def main():
    train_json = "/Users/njany/Documents/instasun-project/detectron2-service/data/test/data/coco_train_transformed.json"
    val_json = "/Users/njany/Documents/instasun-project/detectron2-service/data/test/data/coco_val_transformed.json"
    
    print("Validating training annotations...")
    validate_annotations(train_json)
    print("Validating validation annotations...")
    validate_annotations(val_json)

if __name__ == "__main__":
    main()
