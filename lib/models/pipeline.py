import torch
import os
from sklearn.metrics import classification_report, accuracy_score
import tqdm
import pickle as pkl

from lib.models.linear_classifier import LinearClassifier
from lib.models.gnn import GNN
from lib.dataset import DrivingDataset

################################################################
# ALGORITHM:
# 1. Get SGEmbeddings from the GNN
# 2. Use ImageEmbeddings, SGEmbeddings and AAGEmbeddings in the linear classifier
#
#   SGEmbeddings = TANH(GCNLayer(SG))
#   LinearClassifier = FFNN2(
#               CONCATENATION(
#                   FFNN1(ImageEmbeddings),
#                   SGEmbeddings,
#                   AAGEmbeddings
#                   )
#               )
###############################################################


class CombinerModel(torch.nn.Module):
    def __init__(self, numSceneGraphFeatures: int, sceneGraphEmbeddingSize: int, imgEmbeddingSize: int, reducedImgEmbeddingSize: int, encoderHiddenLayers: list[int], numClasses: int, n_peripheralInputs: int, feedForwardHiddenLayers: list[int]) -> None:
        """
        Class for combining the GNN model with the LinearClassifier for training.
        """
        super().__init__()

        # Initialize the GNN for processing scene graphs
        self.sceneGraphBlock = GNN(
            numSceneGraphFeatures, sceneGraphEmbeddingSize)

        # Initialize the LinearClassifier for combining image, scene graph, and peripheral inputs
        self.ffnnClassifierBlock = LinearClassifier(imgEmbeddingSize, reducedImgEmbeddingSize, encoderHiddenLayers,
                                                    sceneGraphEmbeddingSize, numClasses, n_peripheralInputs, feedForwardHiddenLayers)

        # Create a string representation of model parameters for saving purposes
        self.parameterString = f"{numSceneGraphFeatures}-{sceneGraphEmbeddingSize}-{imgEmbeddingSize}-{reducedImgEmbeddingSize}-{encoderHiddenLayers}-{numClasses}-{n_peripheralInputs}-{feedForwardHiddenLayers}"

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, sceneGraph, imgEmbedding, peripheralInputs):
        # Process the scene graph to get its embedding
        sceneGraphEmbedding = self.sceneGraphBlock(
            sceneGraph.x, sceneGraph.edge_index)

        # Pass the image embedding, scene graph embedding, and peripheral inputs to the classifier
        return self.ffnnClassifierBlock(imgEmbedding, sceneGraphEmbedding, peripheralInputs)

    def train(self, drivingDataset: DrivingDataset, devDataset: DrivingDataset, lr: float, epochs: int) -> None:
        # Create a DataLoader for the training dataset with batch size of 1
        dataLoader = torch.utils.data.DataLoader(
            drivingDataset, batch_size=1, shuffle=True, collate_fn=drivingDataset.customCollate)

        # Set up the optimizer and loss criterion
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        # Dictionary to store metrics for each epoch
        metrics = {}

        # Move the model to the appropriate device (GPU or CPU)
        self.to(self.device)
        self.ffnnClassifierBlock.to(self.device)
        self.sceneGraphBlock.to(self.device)

        for epoch in range(epochs):
            runningTrainLoss = 0
            runningDevLoss = 0
            i = 0
            for item in tqdm.tqdm(dataLoader, desc="Training", leave=True):
                # Extract input (X) and label (y) from the batch
                X, y = item[0]
                optimizer.zero_grad()

                # Move the inputs to the appropriate device
                X[0] = X[0].to(self.device)
                X[1] = X[1].to(self.device)
                X[2] = X[2].to(self.device)
                output = self(X[0], X[1], X[2])

                y = y.to(self.device)
                loss = criterion(output, y)  # Calculate the loss

                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                runningTrainLoss += loss.item()  # Accumulate the training loss

                # Compute the development loss for the same number of items as in the dev dataset
                if i < len(devDataset):
                    with torch.no_grad():
                        X_dev = devDataset.X[i]
                        X_dev[0] = X_dev[0].to(self.device)
                        X_dev[1] = X_dev[1].to(self.device)
                        X_dev[2] = X_dev[2].to(self.device)
                        output_dev = self(X_dev[0], X_dev[1], X_dev[2])
                        y_dev = devDataset.y[i].to(self.device)
                    runningDevLoss += criterion(output_dev, y_dev).item()

                i += 1

            runningTrainLoss /= len(dataLoader)  # Average the training loss
            runningDevLoss /= len(devDataset.X)  # Average the development loss

            # Store the metrics for this epoch
            metrics[epoch] = {'trainMetrics': {}}
            metrics[epoch]['devMetrics'] = self.evaluate(devDataset)
            metrics[epoch]['trainMetrics']['loss'] = runningTrainLoss
            metrics[epoch]['devMetrics']['loss'] = runningDevLoss

            # Print the training and development loss and accuracy for this epoch
            print(
                f"Epoch {epoch+1} | Train loss: {runningTrainLoss:.6f} | Dev loss: {runningDevLoss:.6f} | Dev accuracy: {metrics[epoch]['devMetrics']['accuracy']:.3f}")

            self.metrics = metrics

            if epoch % 5 == 0:
                self.saveModel(epoch)  # Save the model every 5 epochs
                print(f"Model saved at epoch {epoch}")
                print("Computing train metrics...")
                metrics[epoch]['trainMetrics'] = self.evaluate(
                    drivingDataset)  # Evaluate the model on the training set

    def saveModel(self, epoch) -> None:
        # Ensure the cache directory exists
        path = f'./cache'
        if not os.path.exists(path):
            os.makedirs(path)

        # Create a directory for this model based on its parameters
        modelDirName = f'CombinerModel_{self.parameterString}'
        if not os.path.exists(os.path.join(path, modelDirName)):
            os.makedirs(os.path.join(path, modelDirName))

        # Save the model state dictionary
        torch.save(self.state_dict(), os.path.join(
            path, modelDirName, f'model_{epoch}.pt'))

        # Save the metrics
        pkl.dump(self.metrics, open(os.path.join(
            path, f'metrics_{epoch}.pkl'), 'wb'))

    def evaluate(self, dataset: DrivingDataset) -> dict:
        """
        Returns a dictionary of metrics over the given dataset.
        """
        metricsDict = {}

        # List to store predictions
        preds = []
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for X in dataset.X:
                output = self(X[0], X[1], X[2])
                preds.append(output.argmax().item())  # Get the predicted class

        preds = torch.tensor(preds)  # Convert predictions to a tensor

        metricsDict['preds'] = preds

        # Generate a classification report
        metricsDict['report'] = classification_report(
            dataset.y, preds, zero_division=0)

        # Calculate the accuracy score
        metricsDict['accuracy'] = accuracy_score(dataset.y, preds)

        return metricsDict
