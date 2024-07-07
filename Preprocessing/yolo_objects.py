import numpy as np
import pandas as pd
import cv2
import glob, json, os

from sklearn.utils import shuffle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore')

from ultralytics import YOLO
import PIL 
from PIL import Image
from IPython.display import display
import os 
import pathlib 

model = YOLO("yolov8m.pt")

# Function to calculate the center coordinates of a bounding box
def get_center_coordinates(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

# Function to get object detection information from a sequence of images in a specified folder
def get_object_info(videoName):
    # Ensure videoName is enterred
    if videoName is None:
        raise AssertionError("Enter a videoName")
    
    # Get all the image paths
    imagePaths = glob.glob(f"./Datasets/annotatedvideosv1/AnnotatedVideos/{videoName}/*.jpg")
    print(f"Number of images: {len(imagePaths)}")

    # Get pose of each person in each image
    objectDict = {}
    for i, imagePath in enumerate(imagePaths):
        objectDict[imagePath] = []
        print(f"Running inference on image {i+1} of {len(imagePaths)}")
        # print(f"imagePath = {imagePath}")
        results=model.predict(source=imagePath,
              save=True, conf=0.001,iou=0.5)
        
        result = results[0]
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            center = get_center_coordinates(cords)
            conf = round(box.conf[0].item(), 2)
            print("Object type:", class_id)
            print("Coordinates:", cords)
            print("Center:",center)
            print("Probability:", conf)
            print("---")

            objectDict[imagePath].append({"class_id": class_id, "cords": cords, "center": center, "conf": conf})

    # Save the scene graphs in a json file
    os.makedirs("./objects/", exist_ok=True)
    with open(f"./objects/{videoName}.json", "w") as f:
        json.dump(objectDict, f)

    return objectDict

def object_details(imagePath):
    # return object info for a single image
    results=model.predict(source=imagePath,
              save=True, conf=0.001,iou=0.5)
    result = results[0]
    objectDict = []
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        center = get_center_coordinates(cords)
        conf = round(box.conf[0].item(), 2)
        objectDict.append({"class_id": class_id, "cords": cords, "center": center, "conf": conf})
    return objectDict

user_directories = [
    "Dashboard_user_id_24026_3",
    "Dashboard_user_id_24491_0",
    "Dashboard_user_id_35133_0",
    "Dashboard_user_id_38058_0",
    "Dashboard_user_id_49381_0"
]
for user_directory in user_directories:
    get_object_info(user_directory)