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

def get_center_coordinates(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

def get_object_info(videoName="test_video_"):
    # Get all the image paths
    imagePaths = glob.glob(f"./sampledFrames/{videoName}/*.jpg")
    print(f"Number of images: {len(imagePaths)}")

    # Get pose of each person in each image
    objectDict = {}
    for i, imagePath in enumerate(imagePaths):
        objectDict[imagePath] = []
        print(f"Running inference on image {i+1} of {len(imagePaths)}")
    
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

        # break

    # Save the scene graphs in a json file
    os.makedirs("./objects/", exist_ok=True)
    with open("./objects/objects.json", "w") as f:
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


get_object_info()