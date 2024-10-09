import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import yaml
from yaml.loader import SafeLoader

# Function to perform object detection and distance calculation
def detect_objects():
# Load yaml
    with open('custom.yaml',mode='r') as f:
        custom_yaml = yaml.load(f,Loader=SafeLoader)

    labels = custom_yaml['names']
    print(labels)

    #Load yolo model
    yolo = cv2.dnn.readNetFromONNX("C:\\Users\\Nahid\\Desktop\\car detects\\exp\weights\\best.onnx")
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load the image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
      img = cv2.imread(file_path)

    # Check if the image was loaded successfully
    if img is not None:
        image = img.copy()  # Assign the loaded image to 'image'
        # Resize the image
        target_width = 1020
        target_height = 780
        resized_image = cv2.resize(image, (target_width, target_height))
        
    row, col, d = image.shape

    #get the yolo prediction from the image
    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # step-2: get prediction from square array
    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB=True,crop=False)
    yolo.setInput(blob)
    preds = yolo.forward() #prediction from yolo

    #non maximum supression
    # Step 1 
    detections = preds[0]
    boxes = []
    confidences = []
    classes = []

    # Width and Height of the image
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/ INPUT_WH_YOLO
    y_factor = image_h/ INPUT_WH_YOLO


    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] #confidence of detection an object
        if confidence > 0.4:
            class_score = row[5:].max() #maximum probability of object
            class_id = row[5:].argmax()

            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                # construct bounding from the values
                # left, top, width & height
                left = int((cx - 0.5*w)*x_factor)
                top = int((cy - 0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)

                box = np.array([left,top,width,height])

                # append values into the list
                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)

    # Clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()


    # Calculate the average width and height
    if len(boxes_np) > 0:
        total_width = 0
        total_height = 0

        # Calculate total width and height
        for box in boxes_np:
            total_width += box[2]  # Index 2 contains width
            total_height += box[3]  # Index 3 contains height

        # Calculate average width and height
        avg_width = total_width / len(boxes_np)
        avg_height = total_height / len(boxes_np)

        print(f"Average Width: {avg_width} pixels")
        print(f"Average Height: {avg_height} pixels")
    else:
        print("No bounding boxes after NMS.")


    # Sort indices based on confidence
    sorted_indices = sorted(index, key=lambda i: confidences_np[i], reverse=True)

    # Keep track of the indices of the top two detections
    top_indices = sorted_indices[:2]

    # Draw the Bounding Boxes for the top two detections
    for ind in top_indices:
        x, y, w, h = boxes_np[ind]
        bb_conf = int(confidences_np[ind] * 100)
        classes_id = classes[ind]
        class_name = labels[classes_id]

        # Create text to display class name and confidence
        text = f'{class_name}: {bb_conf}%'

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw a filled rectangle as a background for the text
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)

        # Put the text on the image
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)


    # Assuming you have the indices of the top two detections from earlier
    top_indices = sorted_indices[:2]  # Indices of the top two detections after sorting

    # Get the coordinates of the top two bounding boxes
    box1 = boxes_np[top_indices[0]]  # Coordinates of the first box
    box2 = boxes_np[top_indices[1]]  # Coordinates of the second box

    center1 = (box1[0] + box1[2] / 2, box1[1] + box1[3] / 2)
    center2 = (box2[0] + box2[2] / 2, box2[1] + box2[3] / 2)

    # Calculate the distance between the centers of the boxes
    distance_between_centers = ((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)**0.5

    # Assuming car_width_pixels represents the width of the car in pixels
    car_width_pixels = avg_width  # Replace with the actual width of the car in pixels

    # Assuming space_between_boxes_pixels represents the space between two bounding boxes
    space_between_boxes_pixels = distance_between_centers - (box1[2] / 2 + box2[2] / 2)

    # Set a safety margin (e.g., 10% of car width) for clearance
    safety_margin_pixels = 0.1 * car_width_pixels

    # Check if the car can fit between the boxes
    if car_width_pixels + safety_margin_pixels <= space_between_boxes_pixels:
        print("The car can potentially fit between the boxes.")
    else:
        print("The space may not be sufficient for the car to park.")
        
    # Display the image with bounding boxes and distance in Tkinter window
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    panel.config(image=img_tk)
    panel.image = img_tk

# Function to open an image using a file dialog
# (Your existing code for this function)
    



# Create a Tkinter window
root = tk.Tk()
root.title("Object Detection and Distance Calculation")

# Create a button to open an image
open_button = tk.Button(root, text="Open Image", command=detect_objects)  # Call detect_objects directly
open_button.pack()

# Create a panel to display the loaded image
panel = tk.Label(root)
panel.pack()

# Run the Tkinter main loop
root.mainloop()
