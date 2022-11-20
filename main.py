import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import model
# from DatasetGenerator import BreaKHis_Dataset
from DatasetGenerator_Matlab_Handoff import BreaKHis_Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
import time
from tqdm import tqdm
import config
from torch.autograd import Variable

def Experiment():
    start = time.time()

    ##### PRE-TRAINING SETUP #####

    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    print('using device:', device)

    ### Constructing data handlers ###
    print('Loading training data.')
    train_dataset = BreaKHis_Dataset(datasetID=config.datasetID,dsize=config.trainSize,saveDataset=True)
    train_dataset.stackTensor()
    train_DL = DataLoader(train_dataset, batch_size=config.batchSize)
    print('Loading testing data.')
    if config.useValidation:
        test_dataset = BreaKHis_Dataset(datasetID=config.datasetID,dsize=config.trainSize,saveDataset=True)
        test_dataset.stackTensor()
        test_DL = DataLoader(test_dataset, batch_size=config.batchSize)


    print('Time of dataset completion: {:.2f}\nDataset ID: {}'.format(time.time()-start, config.datasetID))

    ### Constructing NN ###
    myNN = model.DCNN(channelsIn=config.numChannels, filtSize=config.imDims)
    if config.modelNum != 000:
        myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format(config.savedModelsPath, config.modelNum)))
        myNN.modelId = config.modelNum
        print('Loaded model num: {}'.format(myNN.modelId))
    else:
        config.modelNum = myNN.modelId
        config.experimentFolder = '/Dataset_{}_Model_{}/'.format(config.datasetID, config.modelNum)
        print('Model generated. Model ID: {}'.format(myNN.modelId))

    if config.showSummary:
        summary(myNN, (config.numChannels, config.imDims, config.imDims))
    myNN.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(myNN.parameters(), lr=config.learningRate, betas=(.9, .999), amsgrad=config.AMSGRAD)

    scaler = None
    if config.halfSize:
        scaler = torch.cuda.amp.GradScaler()

    ### training metrics ###
    trainLoss = []
    trainAccuracy = []
    validationLoss = []
    # bestModel = myNN.state_dict()
    torch.save(myNN.state_dict(), '{}NN_StateDict_{}.pt'.format('./savedModels/', myNN.modelId))
    config.modelNum = myNN.modelId
    config.experimentFolder = '/Dataset_{}_Model_{}/'.format(config.datasetID, config.modelNum)
    bestLoss = 10e10

    ### training routine ###
    initialSummary = True
    if config.showSummary:
        print(torch.cuda.memory_summary())
        print(torch.cuda.memory_snapshot())

    ##### TRAINING ROUTINE #####

    for epoch in range(config.num_epochs):
        myNN.train()
        trainEpochLoss = 0
        n_correct = 0
        ### train batch training ###
        n_batches = 0
        for im_tup in tqdm(train_DL, desc="Batches"):
            n_batches += 1
            im, labels = im_tup[0], Variable(im_tup[1])
            input = Variable(im.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()

            if config.halfSize:
                with torch.cuda.amp.autocast():
                    out = myNN(input)
                    out = out.flatten()
                    trainBatchLoss = criterion(out, labels)
                    scaler.scale(trainBatchLoss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(myNN.parameters(), config.max_norm)

            else:
                out = myNN(input)
                out = out.flatten()
                labels = labels.type(dtype=out.dtype)
                trainBatchLoss = criterion(out,labels)
                trainBatchLoss.backward()
                out_thresh = [1 if x >= 0 else 0 for x in out.cpu()]
                n_correct += sum([1 if z[0] == z[1] else 0 for z in zip(out_thresh,labels.cpu())])


            if config.halfSize:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            trainEpochLoss += float(trainBatchLoss.data)


        # scheduler.step()
        trainLoss.append(trainEpochLoss / n_batches)
        trainAccuracy.append(n_correct / len(train_dataset) * 100)

        if config.useValidation:
            ### validation batch testing ###
            myNN.eval()
            with torch.no_grad():
                valEpochLoss = 0
                n_batches = 0
                for im_tup in tqdm(test_DL, desc="Batches", position=1):
                    n_batches += 1
                    im, label = im_tup[0], im_tup[1]
                    input = im.to(device)
                    label = label.to(device)

                    out = myNN(input)
                    valBatchLoss = criterion(out, label)
                    valEpochLoss += float(valBatchLoss)
                    del input, label, out
                validationLoss.append(valEpochLoss / n_batches)
        else:
            validationLoss.append(0)

        ### store best model ###
        if trainLoss[-1] < bestLoss:
            bestLoss = trainLoss[-1]
            # bestModel = myNN.state_dict()
            torch.save(myNN.state_dict(), '{}NN_StateDict_{}.pt'.format('./savedModels/', myNN.modelId))

        print('{}/{} epochs completed. Train loss: {:.4f}, validation loss: {:.4f}'.format(epoch+1,config.num_epochs,
                                                                                   float(trainLoss[-1]),
                                                                                   float(validationLoss[-1])))
        print('Train accuracy: {}'.format(trainAccuracy[-1]))

    print('done')
    print('Time at training completion: {:.2f}'.format(time.time()-start))


    ##### POST-TRAINING ROUTINE #####

    myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format('./savedModels/',myNN.modelId)))

    ### Observing Results ###

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(trainLoss, label='Train Loss')
    # ax1.plot(validationLoss, label='Validation Loss')
    plt.legend(loc='upper right')
    ax1.set_title('Loss')
    ax2.plot(trainAccuracy, label='Train Accuracy')
    # ax2.plot(validationAccuracy, label='Validation Accuracy')
    ax2.set_title('Accuracy')
    f.suptitle('Model ID: {}, Dataset ID {}'.format(myNN.modelId,config.datasetID))
    plt.show()


    print('Time to completion: {:.2f}'.format(time.time()-start))
    print('Training Complete. Dataset num: {}, Model num: {}'.format(config.datasetID,config.modelNum))


if __name__ == '__main__':
    '''
    sys.argv[1:] : Array of specifications of Experiment(s),
    numChannels imSize dSize modelNum datasetID numChannels imSize dSize modelNum datasetID ...
    ^______________Experiment_Specs_________^ ^______________Experiment_Specs_________^
    '''
    if len(sys.argv)>1:
        numParams = 5
        assert (len(sys.argv)-1)%numParams == 0, 'Incomplete experiment spec set!\n'
        args = sys.argv[1:]
        argsList = [int(arg) for arg in sys.argv[1:]]
        experimentList = list(range(1,len(argsList),numParams))
        for experiment, idx in enumerate(experimentList):
            specs = [argsList[i+experiment*numParams] for i in range(numParams)]
            print('Running Experiment {} with {} num angles, {} imSize, {} dSize, {} modelNum, {} datasetID'.format(experiment+1,specs[0],(specs[1],specs[1]),specs[2],specs[3],specs[4]))
            config.numChannels = specs[0]
            config.anglesFolder = '/nAngles_{}/'.format(config.numChannels)
            config.imDims = specs[1]
            config.dimFolder = '/imSize_{}/'.format(config.imDims)
            config.trainSize = specs[2]
            config.testSize = int(config.trainSize * .2)
            config.modelNum = specs[3]
            config.datasetID = specs[4]
            config.experimentFolder = '/Dataset_{}_Model_{}/'.format(config.datasetID, config.modelNum)
            Experiment()
    else:
        print('Running experiment with config.py specifications...')
        Experiment()