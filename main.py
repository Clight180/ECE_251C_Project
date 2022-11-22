import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from DatasetGenerator_DWT import BreaKHis_DS_DWT
from DatasetGenerator_NoDWT import BreaKHis_DS_NoDWT
from torch.utils.data import DataLoader
import torchsummary as ts
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
    if config.DWT_Input:
        total_dataset = BreaKHis_DS_DWT(datasetID=config.datasetID)
    else:
        total_dataset = BreaKHis_DS_NoDWT(datasetID=config.datasetID)

    print('Splitting dataset into train and validation sets.')
    benigns_idx, malignants_idx = total_dataset.classKeys()
    train_b_idx = int(.8*len(benigns_idx))
    train_m_idx = int(.8*len(malignants_idx))
    train_benigns_idx, test_benigns_idx = torch.utils.data.random_split(benigns_idx, [train_b_idx,len(benigns_idx)-train_b_idx])
    train_malignants_idx, test_malignants_idx = torch.utils.data.random_split(malignants_idx, [train_m_idx,len(malignants_idx)-train_m_idx])
    train_dataset = torch.utils.data.Subset(total_dataset,list(train_benigns_idx) + list(train_malignants_idx))
    test_dataset = torch.utils.data.Subset(total_dataset, list(test_benigns_idx) + list(test_malignants_idx))

    train_DL = DataLoader(train_dataset, batch_size=config.batchSize, shuffle=True)
    test_DL = DataLoader(test_dataset, shuffle=True)

    print("Dataloaders created. \nTrain set size: {} \nTest set size: {}".format(len(train_DL.dataset), len(test_DL.dataset)))


    print('Time of dataset completion: {:.2f}\nDataset ID: {}'.format(time.time()-start, config.datasetID))

    ### Constructing NN ###
    myNN = config.model

    if config.modelNum != 000:
        myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format(config.savedModelsPath, config.modelNum)))
        myNN.modelId = config.modelNum
        print('Loaded model num: {}'.format(myNN.modelId))
    else:
        config.modelNum = myNN.modelId
        config.experimentFolder = '/Dataset_{}_Model_{}/'.format(config.datasetID, config.modelNum)
        print('Model generated. Model ID: {}'.format(myNN.modelId))

    myNN.to(device)
    if config.showSummary:
        ts.summary(myNN, (config.numChannels, config.imDims, config.imDims))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(myNN.parameters(), lr=config.learningRate, betas=(.9, .999), amsgrad=config.AMSGRAD)

    # scaler = None
    # if config.halfSize:
    #     scaler = torch.cuda.amp.GradScaler()

    ### training metrics ###
    trainLoss = []
    trainAccuracy = []
    validationLoss = []
    validationAccuracy = []
    # bestModel = myNN.state_dict()
    torch.save(myNN.state_dict(), '{}NN_StateDict_{}.pt'.format('./savedModels/', myNN.modelId))
    config.modelNum = myNN.modelId
    config.experimentFolder = '/Dataset_{}_Model_{}/'.format(config.datasetID, config.modelNum)
    bestLoss = 10e10

    ### training routine ###
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
        time.sleep(.01)
        for im_tup in tqdm(train_DL, desc="Batches"):
            time.sleep(.01)
            n_batches += 1
            im, labels = im_tup[0], Variable(im_tup[1])
            input = Variable(im.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()

            if config.halfSize:
                with torch.cuda.amp.autocast():
                    out = myNN(input)
                    out = out.flatten()
                    labels = labels.type(dtype=out.dtype)
                    trainBatchLoss = criterion(out, labels)
                    trainBatchLoss.backward()
                    # scaler.scale(trainBatchLoss).backward()
                    # scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(myNN.parameters(), config.max_norm)
                    out_thresh = [1 if x >= 0 else 0 for x in out.cpu()]
                    n_correct += sum([1 if z[0] == z[1] else 0 for z in zip(out_thresh, labels.cpu())])

            else:
                out = myNN(input)
                out = out.flatten()
                labels = labels.type(dtype=out.dtype)
                trainBatchLoss = criterion(out,labels)
                trainBatchLoss.backward()
                out_thresh = [1 if x >= 0 else 0 for x in out.cpu()]
                n_correct += sum([1 if z[0] == z[1] else 0 for z in zip(out_thresh,labels.cpu())])

            del(out)
            del(labels)
            torch.cuda.empty_cache()

            # if config.halfSize:
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            #     optimizer.step()
            optimizer.step()

            trainEpochLoss += float(trainBatchLoss.data)


        # scheduler.step()
        trainLoss.append(trainEpochLoss / n_batches)
        trainAccuracy.append(n_correct / len(train_dataset) * 100)

        ### validation batch testing ###
        myNN.eval()
        with torch.no_grad():
            valEpochLoss = 0
            n_batches = 0
            n_correct = 0
            time.sleep(.01)
            for im_tup in tqdm(test_DL, desc="Validation testing"):
                time.sleep(.01)
                n_batches += 1
                im, labels = im_tup[0], Variable(im_tup[1])
                input = Variable(im.to(device))
                labels = Variable(labels.to(device))

                out = myNN(input)
                out = out.flatten()
                labels = labels.type(dtype=out.dtype)
                testBatchLoss = criterion(out, labels)
                out_thresh = [1 if x >= 0 else 0 for x in out.cpu()]
                n_correct += sum([1 if z[0] == z[1] else 0 for z in zip(out_thresh, labels.cpu())])
                valEpochLoss += float(testBatchLoss.data)
            validationLoss.append(valEpochLoss / n_batches)
            validationAccuracy.append(n_correct / len(test_dataset) * 100)


        ### store best model ###
        if trainLoss[-1] < bestLoss:
            bestLoss = trainLoss[-1]
            # bestModel = myNN.state_dict()
            torch.save(myNN.state_dict(), '{}NN_StateDict_{}.pt'.format('./savedModels/', myNN.modelId))

        print('{}/{} epochs completed. Train loss: {:.4f}, validation loss: {:.4f}'.format(epoch+1,config.num_epochs,
                                                                                   float(trainLoss[-1]),
                                                                                   float(validationLoss[-1])))
        print('Train accuracy: {}, Test accuracy: {}'.format(trainAccuracy[-1], validationAccuracy[-1]))

    print('done')
    print('Time at training completion: {:.2f}'.format(time.time()-start))


    ##### POST-TRAINING ROUTINE #####

    myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format('./savedModels/',myNN.modelId)))

    ### Observing Results ###

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    x = torch.linspace(1, config.num_epochs, steps=config.num_epochs)
    ax1.plot(x, trainLoss, label='Train Loss')
    ax1.plot(x, validationLoss, label='Validation Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('Loss')
    ax2.plot(x, trainAccuracy, label='Train Accuracy')
    ax2.plot(x, validationAccuracy, label='Validation Accuracy')
    ax2.legend(loc='lower right')
    ax2.set_title('Accuracy')
    f.suptitle('Model: {}, Model ID: {}, Dataset ID {}'.format(myNN.name, myNN.modelId,config.datasetID))
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