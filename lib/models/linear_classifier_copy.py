import torch


class LinearClassifier(torch.nn.Module):
    def __init__(self, imgEmbeddingSize: int, reducedImgEmbeddingSize: int, encoderHiddenLayers: list[int],
                 sceneGraphEmbeddingSize: int, numClasses: int, n_peripheralInputs: int, feedForwardHiddenLayers: list[int]) -> None:
        """
        Args:
            imgEmbeddingSize (int): The size of the image embedding.
            reducedImgEmbeddingSize (int): The size of the reduced image embedding.
            encoderHiddenLayers (list[int]): List of hidden layer sizes for the image embedding encoder.
            sceneGraphEmbeddingSize (int): The size of the scene graph embedding.
            numClasses (int): The number of output classes.
            n_peripheralInputs (int): The number of peripheral inputs.
            feedForwardHiddenLayers (list[int]): List of hidden layer sizes for the feed forward network.

        Returns:
            None
        """
        super().__init__()

        # Store input parameters as class attributes
        self.imgEmbeddingSize = imgEmbeddingSize
        self.reducedImgEmbeddingSize = reducedImgEmbeddingSize
        self.encoderHiddenLayers = encoderHiddenLayers
        self.sceneGraphEmbeddingSize = sceneGraphEmbeddingSize
        self.n_peripheralInputs = n_peripheralInputs
        self.feedForwardHiddenLayers = feedForwardHiddenLayers
        self.numClasses = numClasses

# FOR LOOP 1 which builds FFNN for IMAGE
        # Build the image embedding encoder
        self.imgEmbeddingEncoder = torch.nn.Sequential()
        for i in range(len(self.encoderHiddenLayers)):
            # Create a linear layer for the encoder
            linearLayer = torch.nn.Linear(self.imgEmbeddingSize if i == 0 else self.encoderHiddenLayers[i],
                                          self.reducedImgEmbeddingSize if (i == len(self.encoderHiddenLayers) - 1) else self.encoderHiddenLayers[i+1])
            self.imgEmbeddingEncoder.append(linearLayer)
            if i == len(self.encoderHiddenLayers) - 1:
                # Add activation function for the final layer
                self.imgEmbeddingEncoder.append(torch.nn.Tanh())
                continue
            else:
                # Add activation function for intermediate layers
                self.imgEmbeddingEncoder.append(torch.nn.Tanh())

        # Handle case where there are no hidden layers in the encoder
        if len(self.encoderHiddenLayers) == 0:
            linearLayer = torch.nn.Linear(
                self.imgEmbeddingSize, self.reducedImgEmbeddingSize)
            self.imgEmbeddingEncoder.append(linearLayer)
            self.imgEmbeddingEncoder.append(torch.nn.Tanh())

# FOR LOPP 2 which builds FFNN for SCENE GRAPH
        # Build the feedforward layer for final classification
        self.feedForwardLayer = torch.nn.Sequential()
        n_feedForwardInputs = reducedImgEmbeddingSize + \
            sceneGraphEmbeddingSize + n_peripheralInputs
        for i in range(len(self.feedForwardHiddenLayers)):
            # Create a linear layer for the feedforward network
            linearLayer = torch.nn.Linear(n_feedForwardInputs if i == 0 else self.feedForwardHiddenLayers[i],
                                          self.numClasses if i == len(self.feedForwardHiddenLayers) - 1 else self.feedForwardHiddenLayers[i+1])
            self.feedForwardLayer.append(linearLayer)
            if i == len(self.feedForwardHiddenLayers) - 1:
                # Commented out Softmax activation; might be added later if needed
                pass
            else:
                # Add activation function for intermediate layers
                self.feedForwardLayer.append(torch.nn.Tanh())

        # Handle case where there are no hidden layers in the feedforward network
        if len(self.feedForwardHiddenLayers) == 0:
            linearLayer = torch.nn.Linear(n_feedForwardInputs, self.numClasses)
            self.feedForwardLayer.append(linearLayer)
            # Commented out Softmax activation; might be added later if needed

    def forward(self, imgEmbedding, sceneGraphEmbedding, peripheralInputs):
        ###########################################################
        # STEPS:
        #    1. USE FFNN1 FOR IMAGES
        #    2. CONCATENATE THE FFNN1 OUTPUT WITH THE 2 OTHER INPUTS
        #    3. USE FFNN2 ON THIS VECTOR
        ############################################################
        # Pass the image embedding through the encoder
        imgEmbedding = self.imgEmbeddingEncoder(imgEmbedding)

        # Flatten the scene graph embedding
        sceneGraphEmbedding = sceneGraphEmbedding.reshape(-1)

        # Concatenate the image embedding, scene graph embedding, and peripheral inputs
        input = torch.cat(
            (imgEmbedding, sceneGraphEmbedding, peripheralInputs))

        # Pass the concatenated input through the feedforward network
        return self.feedForwardLayer(input)
