from flask import Flask, request, jsonify, send_file, after_this_request
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import io

app = Flask(__name__)

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./models/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
predictor = DefaultPredictor(cfg)

def find_marker_centroid(image):
    # Convert image to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    
    # Threshold the image to get only red color
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Find the centroid of the largest detected red area (assumed to be the marker)
    M = cv2.moments(mask)
    if M["m00"] == 0: return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)


def masks_to_polygons(masks):
    polygons = []
    for mask in masks:
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Convert the contour points to a list of [x, y] pairs
            polygon = approx.reshape(-1, 2).tolist()
            polygons.append(polygon)
            print(f"Contour points: {contour.ravel().tolist()}")  # Log the raw contour points

    return polygons

def find_closest_polygon(polygons, marker_centroid):
    min_distance = float('inf')
    closest_polygon = None
    for polygon in polygons:
        # Calculate the centroid of the polygon
        M = cv2.moments(np.array(polygon, dtype=np.int32))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = np.sqrt((cx - marker_centroid[0]) ** 2 + (cy - marker_centroid[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_polygon = polygon
    return closest_polygon

def process_image(image_data):
    try:
        # Convert bytes to numpy array and process the image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        predictions = predictor(image)
        marker_centroid = find_marker_centroid(image)
        
        # Filter predictions and find polygons
        confidence_threshold = 0.80
        filtered_masks = predictions["instances"].pred_masks[predictions["instances"].scores > confidence_threshold]
        polygons = masks_to_polygons(filtered_masks)
        
        # Find the closest polygon to the marker
        if marker_centroid:
            closest_polygon = find_closest_polygon(polygons, marker_centroid)
            polygons = [closest_polygon] if closest_polygon else []
        
        # Serialize polygon data to JSON and return
        return jsonify({"polygons": polygons})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process image"}), 500


@app.route('/process', methods=['POST'])
def process_route():
    @after_this_request
    def add_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    try:
        image_file = request.files['image']
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        polygon_data = process_image(image_file.read())
        return polygon_data

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
