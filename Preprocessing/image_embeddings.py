# Run this code in the root directory. Else, change the following:
#       1. DATASET_DIR value needs to be changed.
#       2. model.pth is in root directory. The location needs to be changed.

import glob
import json
import os
from icecream import ic
from torch import nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch
print("Importing libraries...")


DATASET_DIR = "Datasets/annotatedvideosv1/AnnotatedVideos/"


# Load the VGGN model
model = models.vgg16(pretrained=True)
# Modify the classifier
model.classifier[6] = torch.nn.Linear(4096, 18)

# Load model weights from .pth file
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()


def getImageEmbedding(imgPath):
    # Load the image
    image = Image.open(imgPath)

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Generate the embedding vector
    with torch.no_grad():
        features = model.features(input_batch)
        features = torch.flatten(features, 1)
        embedding = model.classifier[:4](features)

    # Convert the tensor to a list
    embedding = embedding.squeeze().tolist()
    return embedding


def getImageEmbeddings(videoName=None):
    # Ensure videoName is enterred
    if videoName is None:
        raise AssertionError("Enter a videoName")
    # Use absolute path
    framesDir = os.path.join(DATASET_DIR, videoName)
    imageEmbeddingsPath = os.path.join(framesDir, "imageEmbeddings.json")
    imagePaths = glob.glob(os.path.join(framesDir, "*.jpg"))

    # print(f"l56 framesDir = {framesDir}")
    # print(f"l57 imageEmbeddingsPath = {imageEmbeddingsPath}")

    print(f"Number of images: {len(imagePaths)}")
    # ic(f"Number of images: {len(imagePaths)}")

    imageEmbeddings = {}
    extractedImageEmbeddingsFile = glob.glob(imageEmbeddingsPath)
    extractedImageEmbeddings = {}
    if extractedImageEmbeddingsFile:
        with open(extractedImageEmbeddingsFile[0], "r") as f:
            extractedImageEmbeddings = json.load(f)
        print(
            f"Loaded {len(extractedImageEmbeddings)} image embeddings from file")
        ic(f"Loaded {len(extractedImageEmbeddings)} image embeddings from file")
    else:
        print("No image embeddings found")
        ic("No image embeddings found")

    imageEmbeddings = extractedImageEmbeddings

    try:
        for i, imagePath in enumerate(imagePaths):
            if imagePath in extractedImageEmbeddings:
                print(f"Embedding for image {i} already exists")
                ic(f"Embedding for image {i} already exists")
                continue

            ic(f"Getting embedding for image {i+1} of {len(imagePaths)}")
            imageEmbeddings[imagePath] = getImageEmbedding(imagePath)

            if i % 100 == 0:
                with open(imageEmbeddingsPath, "w") as f:
                    json.dump(imageEmbeddings, f)
    except Exception as e:
        print("Error in image embedding extraction: ", e)
        print("Saving image embeddings to file")
        # os.makedirs(PATH + "/", exist_ok=True)
        with open(imageEmbeddingsPath, "w") as f:
            json.dump(imageEmbeddings, f)
        print("Image embeddings saved to file")
        return imageEmbeddings

    # os.makedirs(PATH + "/", exist_ok=True)
    # imageEmbeddingsPath = os.path.abspath(imageEmbeddingsPath)
    os.makedirs(os.path.dirname(imageEmbeddingsPath), exist_ok=True)
    with open(imageEmbeddingsPath, "w") as f:
        json.dump(imageEmbeddings, f)

    print(f"Image embeddings saved to file: {imageEmbeddingsPath}")


print("Getting image embeddings...")

# iterate through all the user directories and get the image embeddings and store them in
user_directories = [
    "Dashboard_user_id_24026_3",
    "Dashboard_user_id_24491_0",
    "Dashboard_user_id_35133_0",
    "Dashboard_user_id_38058_0",
    "Dashboard_user_id_49381_0"
]

for user_directory in user_directories:
    getImageEmbeddings(user_directory)
