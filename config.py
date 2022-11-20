import torch

# File handling:
processedImsPath = './processed_images/'
rawImsPath = './raw_images/'
tensorDataPath = './TensorData/'
savedModelsPath = './savedModels/'
savedFigsPath = './saved_figs'


# Feature handling
numChannels = 12 # 36 for 3 color channels, 12 wide wavelet packet
imDims = 120 # square, 460/4
trainSize = 800
testSize = int(trainSize*.2)
modelNum = 000 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 000 # Set to pre-existing dataset to avoid generating a new one, ow 000
showSummary = False
printFigs = True
useValidation = False


# Hyperparameters:
num_epochs = 20
batchSize = 20
learningRate = 1e-6
weightDecay = 1e-5
AMSGRAD = True
LRS_Gamma = .995


# GPU acceleration:
USE_GPU = True
halfSize = False
max_norm = .5
dtype = torch.float16 if halfSize else torch.float32


# Runtime vars
theta = None
dimFolder = '/imSize_{}/'.format(imDims)
anglesFolder = '/nAngles_{}/'.format(numChannels)
experimentFolder = '/Dataset_{}_Model_{}/'.format(datasetID,modelNum)
