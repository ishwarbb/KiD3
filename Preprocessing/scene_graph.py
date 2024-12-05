# Run this file in the root directory
import os
import json
import glob
import sys

# The path below may need to be changed if the code is being run on a different device
sys.path.append(os.path.expanduser('.'))
from RelTR.inference import run_inference

DATASET_DIR = "Datasets/annotatedvideosv1/AnnotatedVideos/"


def get_scene_graphs(videoName):
    # Ensure videoName is enterred
    if videoName is None:
        raise AssertionError("Enter a videoName")
    # Use aboslute path
    framesDir = os.path.join(DATASET_DIR, videoName)
    sceneGraphFilePath = os.path.join(DATASET_DIR, videoName, "sceneGraphs.json")
    imagePaths = glob.glob(os.path.join(framesDir, "*.jpg"))

    print(f"framesDir = {framesDir}")
    print(f"sceneGraphFilePath = {sceneGraphFilePath}")

    # Get all the image paths
    print(f"Number of images: {len(imagePaths)}")

    # Run inference on each image
    sceneGraphs = {}
    extractedSceneGraphsFile = glob.glob(sceneGraphFilePath)
    extractedSceneGraphs = {}
    if extractedSceneGraphsFile:
        with open(extractedSceneGraphsFile[0], "r") as f:
            extractedSceneGraphs = json.load(f)
        print(f"Loaded {len(sceneGraphs)} scene graphs from file")
    else:
        print("No scene graphs found")
    print("Extracted scene graphs length ", len(extractedSceneGraphs))

    sceneGraphs = extractedSceneGraphs
    try:
        for i, imagePath in enumerate(imagePaths):
            if imagePath in extractedSceneGraphs:
                print(f"Scene graph for image {i} already exists")
                continue
            print(f"Running inference on image {i+1} of {len(imagePaths)}")
            sceneGraphs[imagePath] = run_inference(
                imagePath, resume='./RelTR/ckpt/checkpoint0149.pth')
            # with open("./sceneGraphs/sceneGraphs.json", "w") as f:
            #     json.dump(sceneGraphs, f)

    except Exception as e:
        print("Error in scene graph extraction: ", e)
        print("Saving scene graphs to file")
        with open(sceneGraphFilePath, "w") as f:
            json.dump(sceneGraphs, f)
        print("Scene graphs saved to file")
        return sceneGraphs

    # Save the scene graphs in a json file
    os.makedirs(os.path.dirname(sceneGraphFilePath), exist_ok=True)
    with open(sceneGraphFilePath, "w") as f:
        json.dump(sceneGraphs, f)

    print(f"Scene graph embeddings saved to file: {sceneGraphFilePath}")
    return sceneGraphs


def scene_graph_details(imagePath):
    # return scene graph for a single image
    sceneGraph = run_inference(imagePath, resume='ckpt/checkpoint0149.pth')
    return sceneGraph


print("Getting scene graph embeddings...")

# iterate through all the user directories and get the scene graph embeddings and store them in
user_directories = [
    "Dashboard_user_id_24026_3",
    "Dashboard_user_id_24491_0",
    "Dashboard_user_id_35133_0",
    "Dashboard_user_id_38058_0",
    "Dashboard_user_id_49381_0"
]

for user_directory in user_directories:
    get_scene_graphs(user_directory)
