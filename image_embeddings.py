print("Importing libraries...")
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch import nn
from icecream import ic

import os
import json
import glob

PATH = "/scratch/isbb"

def getImageEmbedding(imgPath):
        # Load the image
        image = Image.open(imgPath)

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the VGGN model
        model = models.vgg16(pretrained=True)
        # Modify the classifier
        model = nn.DataParallel(model)

        model.module.classifier[6] = torch.nn.Linear(4096, 18)

        # Load model weights from .pth file
        model.load_state_dict(torch.load('model(1).pth', map_location='cpu'))
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        # Generate the embedding vector
        with torch.no_grad():
            features = model.module.features(input_batch)
            features = torch.flatten(features, 1)
            embedding = model.module.classifier[:4](features)

        # Convert the tensor to a list
        embedding = embedding.squeeze().tolist()
        return embedding

        # return input_batch

def getImageEmbeddings(videoName="test_video_"):
    # Use absolute path
    imagePaths = glob.glob("../../../scratch/isbb/test_video/frames/" + f"*.jpg")

    print(f"Number of images: {len(imagePaths)}")
    ic(f"Number of images: {len(imagePaths)}")

    imageEmbeddings = {}
    extractedImageEmbeddingsFile = glob.glob(PATH+"/imageEmbeddings.json")
    extractedImageEmbeddings = {}
    if extractedImageEmbeddingsFile:
        with open(extractedImageEmbeddingsFile[0], "r") as f:
            extractedImageEmbeddings = json.load(f)
        print(f"Loaded {len(extractedImageEmbeddings)} image embeddings from file")
        ic(f"Loaded {len(extractedImageEmbeddings)} image embeddings from file")
    else:
        print("No image embeddings found")
        ic("No image embeddings found")

    imageEmbeddings = extractedImageEmbeddings

    try : 
        for i, imagePath in enumerate(imagePaths):
            if imagePath in extractedImageEmbeddings:
                print(f"Embedding for image {i} already exists")
                ic(f"Embedding for image {i} already exists")
                continue

            ic(f"Getting embedding for image {i+1} of {len(imagePaths)}")
            imageEmbeddings[imagePath] = getImageEmbedding(imagePath)

            if i % 100 == 0:
                os.makedirs(PATH + "/", exist_ok=True)
                with open(PATH +"/imageEmbeddings.json", "w") as f:
                    json.dump(imageEmbeddings, f)
    except Exception as e:
        print("Error in image embedding extraction: ", e)
        print("Saving image embeddings to file")
        os.makedirs(PATH + "/", exist_ok=True)
        with open(PATH+"/imageEmbeddings.json", "w") as f:
            json.dump(imageEmbeddings, f)
        print("Image embeddings saved to file")
        return imageEmbeddings
    
    os.makedirs(PATH + "/", exist_ok=True)
    with open(PATH +"/imageEmbeddings.json", "w") as f:
        json.dump(imageEmbeddings, f)

    print("Image embeddings saved to file")


print("Getting image embeddings...")
getImageEmbeddings()
