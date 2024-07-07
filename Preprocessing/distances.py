# For every image in sampledFrames, get its object information from pose.json, its SceneGrpah information from sceneGraphs.json, and its pose information from pose.json. Then, calculate the distance between each person and each object in the image. Save the results in a json file.
import json
import os
import numpy as np
import glob
from scipy.spatial import distance

def get_distance_po(pose_point, object_center):
    #poise_point is a list of x and y coordinates
    #object_center is a list of x and y coordinates
    print(pose_point)
    print(object_center)
    return distance.euclidean(pose_point, object_center)

def get_distance_pp(pose_point, pose_point2):
    return distance.euclidean(pose_point, pose_point2)

def angle_from_a_point_makes_with_a_line(point, line_point1, line_point2):
    # point is a list of x and y coordinates
    # line_point1 is a list of x and y coordinates
    # line_point2 is a list of x and y coordinates

    # Get the angle between the line and the point
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
    v1 = np.array(line_point1) - np.array(point)
    v2 = np.array(line_point2) - np.array(point)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle

DATASET_DIR = "./Datasets/annotatedvideosv1/AnnotatedVideos/"
# IMAGES_DIR = "./Datasets/annotatedvideosv1/AnnotatedVideos/Dashboard_user_id_24026_3"
def generate_results_json(videoName):
    # Ensure videoName is enterred
    if videoName is None:
        raise AssertionError("Enter a videoName")
    
    IMAGES_DIR = DATASET_DIR + videoName
    # Load the pose, sceneGraph, and object information
    with open(f"./pose/{videoName}.json", "r") as f:
        poseDict = json.load(f)
        # Truncate an extra dot in the beginning of file names in pose.json
        poseDict = {k[1:]: v for k, v in poseDict.items()} 

    with open(f"./Datasets/annotatedvideosv1/AnnotatedVideos/{videoName}/sceneGraphs.json", "r") as f:
        sceneGraphs = json.load(f)

    with open(f"./objects/{videoName}.json", "r") as f:
        objectDict = json.load(f)

    print(f"length of poseDict = {len(poseDict)}")
    print(f"length of sceneGraphs = {len(sceneGraphs)}")
    print(f"length of objectDict = {len(objectDict)}")

    # Iterate over all the images
    results = {}

    # Load existing results from results.json
    if os.path.exists(f"./results/{videoName}.json"):
        with open(f"./results/{videoName}.json", "r") as f:
            results = json.load(f)

    #Iterate over all images in sampledFrames
    imagePaths = glob.glob(f"{IMAGES_DIR}/*.jpg")
    for i,imagePath in enumerate(imagePaths):
        if imagePath in results:
            print(f"Skipping {imagePath} with image number {i} of {len(imagePaths)}")
            continue
        
        # if imagePath[0] != '.':
        #     imagePath = './' + imagePath
        #Check if this key exists in all three
        print(f"imagePath = {imagePath}")
        
        # NOTE: the path in sceneGraphs is "Datasets/..." 
        # where as in the other two it is "./Datasets/..."
        # Hence, the code here had to be adjusted accordingly
        # the flags are used for checking the path in the 3 dictionaries respectively
        flag1 = 0
        flag2 = 0
        flag3 = 0
      
        if imagePath in poseDict:
            print("in poseDict")
            flag1 = 1
        elif imagePath[1:] in poseDict:
            print("in poseDict (stripped)")
            flag1 = 2
        elif imagePath[2:] in poseDict:
            print("in poseDict (double stripped)")
            flag1 = 3
        else:
            print("not in poseDict")
            flag1 = 0

        if imagePath in sceneGraphs:
            print("in sceneGraphs")
            flag2 = 1
        elif imagePath[1:] in sceneGraphs:
            print("in sceneGraphs (stripped)")
            flag2 = 2
        elif imagePath[2:] in sceneGraphs:
            print("in sceneGraphs (double stripped)")
            flag2 = 3
        else:
            print("not in sceneGraphs")
            flag2 = 0

        if imagePath in objectDict:
            print("in objectDict")
            flag3 = 1
        elif imagePath[1:] in objectDict:
            print("in objectDict (stripped)")
            flag3 = 2
        elif imagePath[2:] in objectDict:
            print("in objectDict (double stripped)")
            flag3 = 3
        else:
            print("not` in objectDict")
            flag3 = 0

        print()


        # if one of the three is not present, then skip
        if flag1 * flag2 * flag3 == 0:
            continue

        print(f"Processing image number {i} of {len(imagePaths)}")

        # Get the pose, sceneGraph, and object information for the image
        
        poseInfo = poseDict[imagePath[flag1 - 1:]]
        sceneGraph = sceneGraphs[imagePath[flag2 - 1:]]
        objectInfo = objectDict[imagePath[flag3 - 1:]]

        # print(f"poseInfo = {poseInfo}")
        # print(f"sceneGraph = {sceneGraph}")
        # print(f"objectInfo = {objectInfo}")
        # print()

        distances = {}

        # Get the distance between eyes and hand
        eye1 = poseInfo["points"]["14"] if "14" in poseInfo["points"] else -1
        eye2 = poseInfo["points"]["15"] if "15" in poseInfo["points"] else -1
        hand1 = poseInfo["points"]["4"] if "4" in poseInfo["points"] else -1
        hand2 = poseInfo["points"]["7"] if "7" in poseInfo["points"] else -1
        distances["eye_hand_1"] = get_distance_po(eye1, hand1) if eye1 != -1 and hand1 != -1 else -1
        distances["eye_hand_2"] = get_distance_po(eye2, hand2) if eye2 != -1 and hand2 != -1 else -1
        distances["eye_hand_cross"] = (get_distance_pp(eye1, hand2) + get_distance_pp(eye2, hand1)) / 2 if eye1 != -1 and hand2 != -1 and eye2 != -1 and hand1 != -1 else -1

        # print(poseInfo["points"])
        nose = poseInfo["points"]["0"] if "0" in poseInfo["points"] else -1
        distances["eyes_nose_angle"] = angle_from_a_point_makes_with_a_line(nose, eye1, eye2) if eye1 != -1 and eye2 != -1 and nose != -1 else -1
        
        # Get the distance between hand and phone_object
        phone_object = None
        for obj in objectInfo:
            if obj["class_id"] == "cell phone":
                phone_object = obj["center"]
                break
        distances["hand_phone"] =  get_distance_po(hand1, phone_object) if phone_object and hand1!= -1 else -1 # added hand1 can not be -1

        bottle_object = None
        for obj in objectInfo:
            if  "bottle" in obj["class_id"]:
                bottle_object = obj["center"]
                break
        distances["hand_bottle"] =  get_distance_po(hand1, bottle_object) if bottle_object and hand1!= -1 else -1 # added hand1 can not be -1


        
        results[imagePath] = distances

        # Save the results to a json file
        os.makedirs("./results/", exist_ok=True)
        with open(f"./results/{videoName}.json", "w") as f:
            json.dump(results, f)

        # break


    # print image paths that are in all three
    print("Image paths in all three:")
    count = 0
    for imagePath in imagePaths:
        if imagePath in sceneGraphs and imagePath in objectDict and imagePath in poseDict:
            count += 1
    print(count)

user_directories = [
    "Dashboard_user_id_24026_3",
    "Dashboard_user_id_24491_0",
    "Dashboard_user_id_35133_0",
    "Dashboard_user_id_38058_0",
    "Dashboard_user_id_49381_0"
]

for user_directory in user_directories:
    generate_results_json(user_directory)
