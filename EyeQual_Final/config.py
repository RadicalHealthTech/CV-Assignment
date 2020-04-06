import os
from src import util

experiment = 1

pathTraining = 'data/train'
pathTesting = 'data/test'
pathExperiment = 'experiments'
pathLog = 'log'
imageFormat = ['jpg']

modelParam = {
    'xflip': True,
    'yflip': True,
    'imageSize': 512,
    'imageDim': 3,
    'nClass': 1,
    'batchSize': 8,
    'filter': 64,
    'numConv': 4,
    'kernel': 3,
    'learningRate' : 10e-4,
    'normalize': True

}


####################### DO NOT CHANGE FROM HERE #########################

pathOutput = os.path.join(pathExperiment, str(experiment))
pathOutputLabel = os.path.join(pathOutput, 'labelPrediction.json')
pathOutputHeatMap = os.path.join(pathOutput, 'heatMap')
pathParam = os.path.join(pathOutput, 'param.json')
pathOutputModel = os.path.join(pathOutput, 'model')

util.check_dir(pathOutput)
util.check_dir(pathOutputHeatMap)
util.check_dir(pathLog)
util.check_dir(pathOutputModel)


