import torch


# File handling:
processedImsPath = './processed_images/'
rawImsPath = './raw_images/'
tensorDataPath = './TensorData/'
savedModelsPath = './savedModels/'
savedFigsPath = './saved_figs'


# Feature handling
DWT_Input = True
model_Choice = 'Basic' # 'Basic', 'PLOSONE'
modelNum = 000 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 000 # Set to pre-existing dataset to avoid generating a new one, ow 000
showSummary = False
printFigs = True
useValidation = False


# Hyperparameters:
num_epochs = 10
batchSize = 5
learningRate = 1e-6
weightDecay = 1e-5
AMSGRAD = True


# GPU acceleration:
USE_GPU = True
halfSize = False
max_norm = .5

if halfSize:
    torch.set_default_dtype(torch.half)

import model_Basic
import model_PLOSONE


# Runtime vars
numChannels = 36 if DWT_Input else 3 # 36 for 3 color channels, 12 wide wavelet packet // 3 if no DWT
imDims = 121 if DWT_Input else 460
dimFolder = '/imSize_{}/'.format(imDims)
anglesFolder = '/nAngles_{}/'.format(numChannels)
experimentFolder = '/Dataset_{}_Model_{}/'.format(datasetID,modelNum)

if model_Choice == 'Basic':
    model = model_Basic.DCNN(channelsIn=numChannels)
else:
    model = model_PLOSONE.DCNN(channelsIn=numChannels)
