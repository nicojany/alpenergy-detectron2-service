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
cfg.MODEL.WEIGHTS = "../models/model_final.pth" 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
predictor = DefaultPredictor(cfg)

def find_image_center(image):
    # Get the dimensions of the image
    height, width, _ = image.shape
    # Calculate the center of the image
    center_x = width // 2
    center_y = height // 2
    return (center_x, center_y)

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

    return polygons

def find_closest_polygon(polygons, center):
    min_distance = float('inf')
    closest_polygon = None
    for polygon in polygons:
        # Calculate the centroid of the polygon
        M = cv2.moments(np.array(polygon, dtype=np.int32))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            distance = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_polygon = polygon
    return closest_polygon

def is_polygon_inside_polygon(inner_polygon, outer_polygon):
    for point in inner_polygon:
        if not is_point_inside_polygon(point, outer_polygon):
            return False
    return True

def is_point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def process_image(image_data):
    try:
        # Convert bytes to numpy array and process the image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        predictions = predictor(image)
        image_center = find_image_center(image)
        
        # Filter predictions by confidence and category
        obstacle_category_id = 0
        roof_category_id = 1  
        confidence_threshold = 0.90

        roof_predictions = predictions["instances"].pred_classes == roof_category_id
        obstacle_predictions = predictions["instances"].pred_classes == obstacle_category_id
        high_confidence = predictions["instances"].scores > confidence_threshold

        filtered_roof_masks = predictions["instances"].pred_masks[roof_predictions & high_confidence]
        filtered_obstacle_masks = predictions["instances"].pred_masks[obstacle_predictions & high_confidence]

        roof_polygons = masks_to_polygons(filtered_roof_masks)
        obstacle_polygons = masks_to_polygons(filtered_obstacle_masks)

        closest_roof_polygon = find_closest_polygon(roof_polygons, image_center)
        roof_polygons = [closest_roof_polygon] if closest_roof_polygon else []

        obstacles_on_roof = []
        if closest_roof_polygon:
            for obstacle in obstacle_polygons:
                if is_polygon_inside_polygon(obstacle, closest_roof_polygon):  
                    obstacles_on_roof.append(obstacle)
        
        print(f"Polygon: {closest_roof_polygon}")  
        print(f"Obstacles: {obstacles_on_roof}")  

        return jsonify({"polygons": closest_roof_polygon, "obstacles": obstacles_on_roof})

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
    app.run(host='0.0.0.0', port=5004)
