from RelTR.inference import run_inference
import glob, json, os

def get_scene_graphs(videoName="test_video_"):


    # Get all the image paths
    imagePaths = glob.glob(f"./sampledFrames/{videoName}/*.jpg")
    print(f"Number of images: {len(imagePaths)}")

    # Run inference on each image
    sceneGraphs = {}
    extractedSceneGraphsFile = glob.glob("./sceneGraphs/sceneGraphs.json")
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
            sceneGraphs[imagePath] = run_inference(imagePath, resume = './RelTR/ckpt/checkpoint0149.pth')
            # with open("./sceneGraphs/sceneGraphs.json", "w") as f:
            #     json.dump(sceneGraphs, f)

    except Exception as e:
        print("Error in scene graph extraction: ", e)
        print("Saving scene graphs to file")
        os.makedirs("./sceneGraphs/", exist_ok=True)
        with open("./sceneGraphs/sceneGraphs.json", "w") as f:
            json.dump(sceneGraphs, f)
        print("Scene graphs saved to file")
        return sceneGraphs

    # Save the scene graphs in a json file
    os.makedirs("./sceneGraphs/", exist_ok=True)
    with open("./sceneGraphs/sceneGraphs.json", "w") as f:
        json.dump(sceneGraphs, f)


    return sceneGraphs

def scene_graph_details(imagePath):
    # return scene graph for a single image
    sceneGraph = run_inference(imagePath, resume = 'ckpt/checkpoint0149.pth')
    return sceneGraph

# print("Running inference on test_video")
# sceneGraphs = get_scene_graphs()
# print(sceneGraphs)

get_scene_graphs("test_video_")
    

