import os
import numpy as np
import pandas as pd
from typing import Literal
import sys
sys.path.append(os.path.expanduser('~/Desktop/Distracted-Driving-Detection'))
from lib.driving_dataset.Preprocessor import Preprocessor

DATASET_DIR = './Datasets/annotatedvideosv1/AnnotatedVideos/'

preproc = Preprocessor()


def saveSplits(trainDf: pd.DataFrame,
               devDf: pd.DataFrame,
               testDf: pd.DataFrame,
               videoName: str,
               type: Literal["imageEmbeddings", "sceneGraphs"]) -> None:
    trainDf.to_json(os.path.join(DATASET_DIR, videoName, type, 'train.json'),
                    orient='index')
    devDf.to_json(os.path.join(DATASET_DIR, videoName, type, 'dev.json'),
                  orient='index')
    testDf.to_json(os.path.join(DATASET_DIR, videoName, type, 'test.json'),
                   orient='index')


def splitData(videoName: str) -> None:
    # load data
    print(DATASET_DIR)
    print(videoName)
    sceneGraphData = pd.read_json(os.path.join(
        DATASET_DIR, videoName, 'sceneGraphs.json'), orient='index')
    imageEmbeddings = pd.read_json(os.path.join(
        DATASET_DIR, videoName, 'imageEmbeddings.json'), orient='index')
    peripheralInputs = pd.read_json("./results/ALL.json", orient='index')


    tempData = sceneGraphData.merge(
        imageEmbeddings, left_index=True, right_index=True)
    print(len(tempData))

    peripheralInputs.index = peripheralInputs.index.map(lambda x: x[2:])

    # print(f"l42 peripiheralInputs = {peripheralInputs}")
    aligned_peripheralInputs = peripheralInputs.reindex(tempData.index)

    # print(f"l44 aligned_peripheralInputs = {aligned_peripheralInputs}")
    allData = tempData.merge(aligned_peripheralInputs, left_index=True, right_index=True)

    # print("l46 allData = ")
    # print(allData)
    # print()

    # # Check the first few indexes of tempData and peripheralInputs to debug potential mismatches
    # print("Indexes of tempData:", tempData.index[:5])
    # print("Indexes of aligned_peripheralInputs:", aligned_peripheralInputs.index[:5])

    sceneGraphData = allData[sceneGraphData.columns]
    imageEmbeddings = allData[imageEmbeddings.columns]
    peripheralInputs = allData[peripheralInputs.columns]

    print("l46")
    print(sceneGraphData.shape)
    print(imageEmbeddings.shape)
    print(peripheralInputs.shape)
    print("l50")

    # split into train, test, dev
    trainSize = 0.6
    devSize = 0.2
    testSize = 1 - trainSize - devSize

    np.random.seed(42)

    # get indices
    indicesSet = set(range(len(sceneGraphData)))

    trainIndices = set(np.random.choice(list(indicesSet), int(
        trainSize * len(sceneGraphData)), replace=False))
    indicesSet.difference_update(trainIndices)

    devIndices = set(np.random.choice(list(indicesSet), int(
        devSize * len(sceneGraphData)), replace=False))
    indicesSet.difference_update(devIndices)

    testIndices = np.array(list(indicesSet))
    trainIndices = np.array(list(trainIndices))
    devIndices = np.array(list(devIndices))

    # get data
    sceneGraphTrain = sceneGraphData.iloc[trainIndices]
    sceneGraphDev = sceneGraphData.iloc[devIndices]
    sceneGraphTest = sceneGraphData.iloc[testIndices]

    imageEmbeddingsTrain = imageEmbeddings.iloc[trainIndices]
    imageEmbeddingsDev = imageEmbeddings.iloc[devIndices]
    imageEmbeddingsTest = imageEmbeddings.iloc[testIndices]

    peripheralTrain = peripheralInputs.iloc[trainIndices]
    peripheralDev = peripheralInputs.iloc[devIndices]
    peripheralTest = peripheralInputs.iloc[testIndices]
    # save
    # save to the "ALL" instead of "ALL/frames"
    # before running the saveSplits part of the code, first create the 2 folders - sceneGraphs and imageEmbeddings in the ALL folder
    saveSplits(sceneGraphTrain, sceneGraphDev,
               sceneGraphTest, "ALL", 'sceneGraphs')
    saveSplits(imageEmbeddingsTrain, imageEmbeddingsDev,
               imageEmbeddingsTest, "ALL", 'imageEmbeddings')

    # save peripheralInputs
    peripheralTrain.to_json('./results/train.json', orient='index')
    peripheralDev.to_json('./results/dev.json', orient='index')
    peripheralTest.to_json('./results/test.json', orient='index')


if __name__ == '__main__':
    splitData('ALL/frames')
