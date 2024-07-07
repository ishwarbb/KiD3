from lib.dataset import DrivingDataset

trainDataset = DrivingDataset(videoName='ALL', split='train')
testDataset = DrivingDataset(videoName='ALL', split='test')
devDataset = DrivingDataset(videoName='ALL', split='dev')

from lib.models.pipeline import CombinerModel

model = model = CombinerModel(trainDataset.data.iloc[0, 0].num_features,
                              sceneGraphEmbeddingSize=64,
                              imgEmbeddingSize=trainDataset.imageEmbeddingSize,
                              reducedImgEmbeddingSize=trainDataset.imageEmbeddingSize,
                              encoderHiddenLayers=[],
                              numClasses=trainDataset.numClasses,
                              n_peripheralInputs=trainDataset.peripheralInputSize,
                              feedForwardHiddenLayers=[128])

model.train(trainDataset, devDataset, epochs=100, lr=0.0005)

import pickle as pkl
import os
def load_metrics(epoch) -> dict:
    path = f'./cache'
    metricsFileName = os.path.join(path, f'metrics_{epoch}.pkl')

    with open(metricsFileName, 'rb') as f:
        metrics = pkl.load(f)

    return metrics

epoch = 10  
metrics = load_metrics(epoch)

accuracy = metrics[10]['devMetrics']['accuracy']
report = metrics[10]['devMetrics']['report']

print(accuracy)
print(report)


