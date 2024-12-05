import os
import glob
import shutil
import pandas as pd

DATASET_DIR = 'Datasets/annotatedvideosv1/AnnotatedVideos'
TARGET_DIR = 'ALL/frames/'

PERIPHERAL_INPUT_DIR = 'results'

user_directories = [
    "Dashboard_user_id_24026_3",
    "Dashboard_user_id_24491_0",
    "Dashboard_user_id_35133_0",
    "Dashboard_user_id_38058_0",
    "Dashboard_user_id_49381_0"
]

runningSceneGraphDfs = []
runningImageEmbeddingDfs = []
runningPeripheralInputDfs = []

# Create ALL directory if it does not exist
ALL_DIR = os.path.join(DATASET_DIR, "ALL")
if not os.path.exists(ALL_DIR):
    os.makedirs(ALL_DIR)
    print("ALL directory created")

# Create ALL/frames if it does not exist
FRAMES_DIR = os.path.join(DATASET_DIR, TARGET_DIR)
if not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)
    print("frames directory created")

# for videoName in os.listdir(DATASET_DIR):
for videoName in user_directories:
    if videoName in TARGET_DIR:
        print("l25 continuing")
        continue

    userId = videoName.split('_')[-2]

    # copy all jpg files to the target directory
    jpgFiles = glob.glob(os.path.join(DATASET_DIR, videoName, '*.jpg'))

    for jpgFile in jpgFiles:
        fileName = jpgFile.split('/')[-1]
        shutil.copy(jpgFile, os.path.join(DATASET_DIR, TARGET_DIR, fileName))

    print(f"DATASET_DIR = {DATASET_DIR}")
    print(f"videoName = {videoName}")

    # read video scene graphs
    sceneGraphs = pd.read_json(os.path.join(
        DATASET_DIR, videoName, 'sceneGraphs.json'), orient='index')
    runningSceneGraphDfs.append(sceneGraphs)

    # read image embeddings
    imageEmbeddings = pd.read_json(os.path.join(
        DATASET_DIR, videoName, 'imageEmbeddings.json'), orient='index')
    runningImageEmbeddingDfs.append(imageEmbeddings)

    # read peripheral inputs
    peripheralInputs = pd.read_json(
        f"./results/{videoName}.json", orient='index')
    runningPeripheralInputDfs.append(peripheralInputs)

# concatenate along rows
concatenatedSceneGraphs = pd.concat(runningSceneGraphDfs, axis=0)
concatenatedSceneGraphs.to_json(os.path.join(
    DATASET_DIR, TARGET_DIR, 'sceneGraphs.json'), orient='index')

concatenatedImageEmbeddings = pd.concat(runningImageEmbeddingDfs, axis=0)
concatenatedImageEmbeddings.to_json(os.path.join(
    DATASET_DIR, TARGET_DIR, 'imageEmbeddings.json'), orient='index')

concatenatedPeripheralInputs = pd.concat(runningPeripheralInputDfs, axis=0)
concatenatedPeripheralInputs.to_json('./results/ALL.json', orient='index')
