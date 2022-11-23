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
DWT_Input = False
model_Choice = 'Experimental' # 'Basic', 'PLOSONE', 'Experimental'
modelNum = 000 # Set to pre-existing model to avoid training from epoch 1 , ow 000
datasetID = 317 # Set to pre-existing dataset to avoid generating a new one, ow 000
datasetSize = -1 # -1 for use whole dataset, ow your choice
showSummary = True
printFigs = True


# Hyperparameters:
num_epochs = 30
batchSize = 300
learningRate = 5e-4
lrs_Gamma = .93
weightDecay = 1e-4
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

