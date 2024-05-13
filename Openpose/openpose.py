import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

import glob, json, os

# print current working directory
import os
print(os.getcwd())

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

import math
# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset, filter):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    points = {}
    
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            if i not in filter:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
            cv2.putText(canvas, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
            points[i] = (int(x), int(y))
    
    return canvas, points


# Iterate through each image in the folder "sampledFrames" and detect the pose
# of the person in each frame


def detect_pose(videoName="test_video_"):
    # Get all the image paths
    imagePaths = glob.glob(f"../sampledFrames/{videoName}/*.jpg")
    print(f"Number of images: {len(imagePaths)}")

    # Get pose of each person in each image
    poseDict = {}
    extractedPoseDictFile = glob.glob(f"./pose/pose.json")
    extractedPoseDict = {}
    if extractedPoseDictFile:
        with open(extractedPoseDictFile[0], "r") as f:
            extractedPoseDict = json.load(f)
        print(f"Loaded {len(extractedPoseDict)} scene graphs from file")
    else:
        print("No scene graphs found")

    poseDict = extractedPoseDict

    try:

        for i, imagePath in enumerate(imagePaths):
            if imagePath in extractedPoseDict.keys():
                print(f"Pose for image {i} already exists")
                continue
            print(f"Running inference on image {i+1} of {len(imagePaths)}")
            oriImg = cv2.imread(imagePath)
            candidate, subset = body_estimation(oriImg)
            # print(candidate)
            # print(subset)
            # filter is [1 ... to 17]
            filter = [1,2,3,4,7,8,9,10,11,12,13,14,15,16,17]
            blank_canvas = np.zeros(oriImg.shape)
            blank_canvas , points = draw_bodypose(blank_canvas, candidate, subset, filter)
            print(points)
            candidate = candidate.tolist()
            subset = subset.tolist()


            poseDict[imagePath] = {
                "candidate" : candidate,    
                "subset" : subset,
                "points" : points
            }

            print(poseDict)

            # break

            # Save the pose in a json file
            os.makedirs("./pose/", exist_ok=True)
            with open("./pose/pose.json", "w") as f:
                print(type(poseDict))
                # write the dictionary to the json file
                json.dump(poseDict, f)

    except Exception as e:
        print("Error in scene graph extraction: ", e)
        print("Saving poseDict to file")
        # Save the pose in a json file
        os.makedirs("./pose/", exist_ok=True)
        with open("./pose/pose.json", "w") as f:
            print(type(poseDict))
            # write the dictionary to the json file
            json.dump(poseDict, f)
        return poseDict

    # Save the scene graphs in a json file
    os.makedirs("./pose/", exist_ok=True)
    with open("./pose/pose.json", "w") as f:
        print(type(poseDict))
        # write the dictionary to the json file
        json.dump(poseDict, f)

    return poseDict

    return poseDict


def pose_details(imagePath):
    # return pose info for a single image
    oriImg = cv2.imread(imagePath)
    candidate, subset = body_estimation(oriImg)
    # print(candidate)
    # print(subset)
    # filter is [1 ... to 17]
    filter = [1,2,3,4,7,8,9,10,11,12,13,14,15,16,17]
    blank_canvas = np.zeros(oriImg.shape)
    blank_canvas , points = draw_bodypose(blank_canvas, candidate, subset, filter)
    print(points)
    candidate = candidate.tolist()
    subset = subset.tolist()
    poseDict = {
        "candidate" : candidate,    
        "subset" : subset,
        "points" : points
    }
    return poseDict

detect_pose()
# pose_details(imagePath="../sampledFrames/test_video/0.jpg")

