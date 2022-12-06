import model_Basic
import model_PLOSONE
import model_Experimental

# File handling:
processedImsPath = './processed_images/'
rawImsPath = './raw_images/'
tensorDataPath = './TensorData/'
savedModelsPath = './savedModels/'
savedFigsPath = './saved_figs'


# Feature handling
DWT_Input = True
model_Choice = 'Experimental' # 'Basic', 'PLOSONE', 'Experimental'
modelNum = 000 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 509 # Set to pre-existing dataset to avoid generating a new one, ow 000
datasetSize = -1 # -1 for use whole dataset, ow your choice
showSummary = True
printFigs = True
NoiseLevelFolder = 'no_noise'
# DWTFolder = 'dwt_haar'
DWTFolder = 'dwt_L2x12_bior4.4'
imTransforms = True

# Hyperparameters:
num_epochs = 200
batchSize = 300
learningRate = 4e-4
lrs_Gamma = .9965
weightDecay = 4e-4
AMSGRAD = True


# GPU acceleration:
USE_GPU = True
halfSize = False
max_norm = .5


# Runtime vars
numChannels = 36 if DWT_Input else 3 # 36 for 3 color channels, 12 wide wavelet packet // 3 if no DWT
imDims = 121 if DWT_Input else 460
dimFolder = '/imSize_{}/'.format(imDims)
anglesFolder = '/nAngles_{}/'.format(numChannels)
experimentFolder = '/Dataset_{}_Model_{}/'.format(datasetID,modelNum)

if model_Choice == 'Basic':
    model = model_Basic.DCNN(channelsIn=numChannels)
elif model_Choice == 'PLOSONE':
    model = model_PLOSONE.DCNN(channelsIn=numChannels)
elif model_Choice == 'Experimental':
    model = model_Experimental.DCNN(channelsIn=numChannels)

