import cv2
import numpy as np
import yaml

class YOLO():
    def __init__(self, onnx_model_path, custom_yaml_path):
        # Load yaml
        with open(custom_yaml_path, mode='r') as f:
            custom_yaml = yaml.safe_load(f)
            
        self.labels = custom_yaml['names']
        self.nc = custom_yaml['nc']

        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model_path)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        row, col, d = image.shape

        # Get the YOLO prediction from the image
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Step 2: Get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # Prediction from YOLO

        # Non-maximum suppression
        # Initialize lists
        boxes = []
        confidences = []
        classes = []

        # Width and Height of the image
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(preds)):
            row = preds[i]
            confidence = row[4]  # Confidence of detecting an object
            if confidence > 0.4:
                class_score = row[5:].max()  # Maximum probability of object
                class_id = row[5:].argmax()

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # Construct bounding box coordinates
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    # Append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # NMS
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()

        # Draw the bounding boxes
        for ind in index:
            x, y, w, h = boxes_np[ind]
            bb_conf = int(confidences_np[ind] * 100)
            class_id = classes[ind]
            class_name = self.labels[class_id]
            colors = self.generate_colors(class_id)

            text = f'{class_name}: {bb_conf}%'

            cv2.rectangle(image, (x, y), (x+w, y+h), colors, 2)
            cv2.rectangle(image, (x, y-30), (x+w, y), colors, -1)
            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

        return image

    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])
