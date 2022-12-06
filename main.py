import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from DatasetGenerator_DWT import BreaKHis_DS_DWT
from DatasetGenerator_NoDWT import BreaKHis_DS_NoDWT
from torch.utils.data import DataLoader
import torchsummary as ts
import numpy as np
import time
from tqdm import tqdm
import config
from torch.autograd import Variable
import gc
import model_Basic, model_Experimental, model_PLOSONE
import torchvision.transforms as transforms
import torchvision

def Experiment(trainBenignFold=None, trainMaligFold=None, valBenignFold=None, valMaligFold=None, KFC=False):
    start = time.time()
    experimentID = np.random.randint(100, 999)
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

    if config.datasetSize != -1 and not KFC:
        maxSize_singleClass = len(benigns_idx)
        splitQuant = int(config.datasetSize/2)
        if splitQuant >= maxSize_singleClass:
            exit('Class quantity exceeds maximum size...')
        benigns_idx, _ = torch.utils.data.random_split(benigns_idx, [splitQuant, maxSize_singleClass - splitQuant])
        malignants_idx, _ = torch.utils.data.random_split(malignants_idx, [splitQuant, maxSize_singleClass - splitQuant])

    # K-Fold Cross Validation
    if KFC:
        train_benigns_idx = trainBenignFold
        train_malignants_idx = trainMaligFold
        test_benigns_idx = valBenignFold
        test_malignants_idx = valMaligFold
    else:
        train_b_quant = int(.7*len(benigns_idx))
        train_m_quant = int(.7*len(malignants_idx)) # Should be same number but why not
        train_benigns_idx, test_benigns_idx = torch.utils.data.random_split(benigns_idx, [train_b_quant,len(benigns_idx)-train_b_quant])
        train_malignants_idx, test_malignants_idx = torch.utils.data.random_split(malignants_idx, [train_m_quant,len(malignants_idx)-train_m_quant])


    train_dataset = torch.utils.data.Subset(total_dataset,list(train_benigns_idx) + list(train_malignants_idx))
    test_dataset = torch.utils.data.Subset(total_dataset, list(test_benigns_idx) + list(test_malignants_idx))

    train_DL = DataLoader(train_dataset, batch_size=config.batchSize, shuffle=True)
    test_DL = DataLoader(test_dataset, batch_size=config.batchSize, shuffle=True)

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


    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(myNN.parameters(), lr=config.learningRate, betas=(.9, .999), weight_decay=config.weightDecay, amsgrad=config.AMSGRAD)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lrs_Gamma)

    scaler = None
    if config.halfSize:
        scaler = torch.cuda.amp.GradScaler()

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

    myNN.to(device)

    if config.showSummary:
        ts.summary(myNN, (config.numChannels, config.imDims, config.imDims))
        print(torch.cuda.memory_summary())
        print(torch.cuda.memory_snapshot())

    ##### TRAINING ROUTINE #####

    for epoch in range(config.num_epochs):

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
                out_thresh = [1 if x >= 0 else 0 for x in out.detach()]
                n_correct += sum([1 if z[0] == z[1] else 0 for z in zip(out_thresh, labels.detach())])
                valEpochLoss += float(testBatchLoss)
            validationLoss.append(valEpochLoss / n_batches)
            validationAccuracy.append(n_correct / len(test_dataset) * 100)
            del (out)
            del (labels)
            del (testBatchLoss)
            if config.USE_GPU:
                gc.collect()
                torch.cuda.empty_cache()


        myNN.train()

        trainEpochLoss = 0
        n_correct = 0
        ### train batch training ###
        n_batches = 0
        time.sleep(.01)
        for im_tup in tqdm(train_DL, desc="Batches"):
            time.sleep(.01)
            n_batches += 1
            ims, labels = im_tup[0], Variable(im_tup[1])
            ims = ims.to(device)
            labels = labels.to(device)

            if config.imTransforms:
                # transform images
                for idx, im in enumerate(ims):
                    HW = im[0].size()
                    angle = np.random.poisson(lam=15)
                    prob = .1
                    flipH = np.random.choice([True, False], p=[prob, 1-prob])
                    flipV = np.random.choice([True, False], p=[prob, 1-prob])
                    rotateIm = np.random.choice([True, False], p=[prob, 1-prob])

                    if flipH:
                        im = transforms.functional.hflip(im)
                    if flipV:
                        im = transforms.functional.vflip(im)
                    if rotateIm:
                        im = transforms.functional.rotate(im, angle=angle, expand=True,
                                                         interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
                        crop = transforms.CenterCrop(HW)
                        im = crop(im)
                        ims[idx] = im

            optimizer.zero_grad()

            if config.halfSize:
                im = torch.tensor(ims,dtype=torch.half)
                labels = torch.tensor(labels, dtype=torch.half)
                input = Variable(im.to(device))
                labels = Variable(labels.to(device))
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = myNN(input)
                    out = out.flatten()
                    labels = labels.type(dtype=out.dtype)
                    trainBatchLoss = criterion(out, labels)
                # trainBatchLoss.backward()
                scaler.scale(trainBatchLoss).backward()
                torch.nn.utils.clip_grad_norm_(myNN.parameters(), config.max_norm)
                out_thresh = [1 if x >= 0 else 0 for x in out.cpu()]
                n_correct += sum([1 if z[0] == z[1] else 0 for z in zip(out_thresh, labels.cpu())])

            else:
                input = Variable(ims)
                labels = Variable(labels)
                out = myNN(input)
                out = out.flatten()
                labels = labels.type(dtype=out.dtype)
                trainBatchLoss = criterion(out,labels)
                trainBatchLoss.backward()
                out_thresh = [1 if x >= 0 else 0 for x in out.detach()]
                n_correct += sum([1 if z[0] == z[1] else 0 for z in zip(out_thresh,labels.detach())])


            trainEpochLoss += float(trainBatchLoss)
            del(out)
            del(labels)
            del(input)
            # del(trainBatchLoss) # Don't do this...........
            if config.USE_GPU:
                gc.collect()
                torch.cuda.empty_cache()

            if config.halfSize:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # optimizer.step()


        scheduler.step()
        trainLoss.append(trainEpochLoss / n_batches)
        trainAccuracy.append(n_correct / len(train_dataset) * 100)




        ### store best model ###
        if trainLoss[-1] < bestLoss:
            bestLoss = trainLoss[-1]
            # bestModel = myNN.state_dict()
            torch.save(myNN.state_dict(), '{}NN_StateDict_{}.pt'.format('./savedModels/', myNN.modelId))

        print('{}/{} epochs completed. Train loss: {:.4f}, validation loss: {:.4f}'.format(epoch+1,config.num_epochs,
                                                                                   float(trainLoss[-1]),
                                                                                   float(validationLoss[-1])))
        print('Train accuracy: {:.4f}, Test accuracy: {:.4f}'.format(trainAccuracy[-1], validationAccuracy[-1]))

    print('done')
    print('Time at training completion: {:.2f}'.format(time.time()-start))


    ##### POST-TRAINING ROUTINE #####

    myNN.load_state_dict(torch.load('{}/NN_StateDict_{}.pt'.format('./savedModels/',myNN.modelId)))

    ### Observing Results ###

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    x = torch.linspace(1, config.num_epochs, steps=config.num_epochs)
    ax1.plot(x, trainLoss, label='Train Loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.plot(x, validationLoss, label='Validation Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('Loss')
    ax2.plot(x, trainAccuracy, label='Train Accuracy')
    ax2.plot(x, validationAccuracy, label='Validation Accuracy')
    ax2.legend(loc='lower right')
    ax2.set_title('Accuracy')
    f.suptitle('Exp ID: {}, Dataset ID {}, With DWT: {}'.format(experimentID,config.datasetID, config.DWT_Input))
    plt.show()


    print('Time to completion: {:.2f}'.format(time.time()-start))
    print('Training Complete. Dataset num: {}, Model num: {}'.format(config.datasetID,config.modelNum))
    return trainLoss, trainAccuracy, validationLoss, validationAccuracy

if __name__ == '__main__':

    if len(sys.argv)>1:
        KFC = True
        numK = int(sys.argv[1])
        print('Performing Cross Validation for {} folds'.format(numK))

        if config.datasetID == 000:
            exit('Perform on a pre sorted dataset')

        if config.DWT_Input:
            total_dataset = BreaKHis_DS_DWT(datasetID=config.datasetID)
        else:
            total_dataset = BreaKHis_DS_NoDWT(datasetID=config.datasetID)

        benigns_idx, malignants_idx = total_dataset.classKeys()

        maxSize_singleClass = len(benigns_idx)
        if config.datasetSize != -1:
            splitQuant = int(config.datasetSize / 2)
            if splitQuant >= maxSize_singleClass:
                exit('Class quantity exceeds maximum size...')
        else:
            splitQuant = maxSize_singleClass

        benigns_idx, unused_benings = torch.utils.data.random_split(benigns_idx, [splitQuant, maxSize_singleClass - splitQuant])
        malignants_idx, unused_maligs = torch.utils.data.random_split(malignants_idx, [splitQuant, maxSize_singleClass - splitQuant])

        fold_unit_quant = int(splitQuant / numK)
        CV_TrainLoss = np.array([0]*config.num_epochs)
        CV_TrainAcc = np.array([0]*config.num_epochs)
        CV_ValLoss = np.array([0]*config.num_epochs)
        CV_ValAcc = np.array([0]*config.num_epochs)
        for k_th_fold in range(numK):
            config.modelNum = 000

            if config.model_Choice == 'Basic':
                config.model = model_Basic.DCNN(channelsIn=config.numChannels)
            elif config.model_Choice == 'PLOSONE':
                config.model = model_PLOSONE.DCNN(channelsIn=config.numChannels)
            elif config.model_Choice == 'Experimental':
                config.model = model_Experimental.DCNN(channelsIn=config.numChannels)

            benignFolds = torch.utils.data.random_split(benigns_idx, [fold_unit_quant] * numK + [maxSize_singleClass-len(unused_benings)-fold_unit_quant*numK])
            malignantFolds = torch.utils.data.random_split(malignants_idx, [fold_unit_quant] * numK + [maxSize_singleClass-len(unused_benings)-fold_unit_quant*numK])

            val_b_folds = benignFolds[k_th_fold]
            val_m_folds = malignantFolds[k_th_fold]
            del(benignFolds[k_th_fold])
            del(malignantFolds[k_th_fold])
            train_b_folds = torch.utils.data.ConcatDataset(benignFolds)
            train_m_folds = torch.utils.data.ConcatDataset(malignantFolds)
            args = (train_b_folds, train_m_folds, val_b_folds,val_m_folds, KFC)
            expTrainLoss, expTrainAccuracy, expValidationLoss, expValidationAccuracy = Experiment(args)
            ExpTL = np.array(expTrainLoss)
            ExpTA = np.array(expTrainAccuracy)
            ExpVL = np.array(expValidationLoss)
            ExpVA = np.array(expValidationAccuracy)

            np.savetxt("Train_Loss_{}.csv".format(k_th_fold), ExpTL)
            np.savetxt("Train_Accuracy_{}.csv".format(k_th_fold), ExpTA)
            np.savetxt("Val_Loss_{}.csv".format(k_th_fold), ExpVL)
            np.savetxt("Val_Acc_{}.csv".format(k_th_fold), ExpVA)

            CV_TrainLoss = CV_TrainLoss + ExpTL
            CV_TrainAcc = CV_TrainAcc + ExpTA
            CV_ValLoss = CV_ValLoss + ExpVL
            CV_ValAcc = CV_ValAcc + ExpVA

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        x = torch.linspace(1, config.num_epochs, steps=config.num_epochs)
        ax1.plot(x, CV_TrainLoss/numK, label='Train Loss')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.plot(x, CV_ValLoss/numK, label='Validation Loss')
        ax1.legend(loc='upper right')
        ax1.set_title('Loss')
        ax2.plot(x, CV_TrainAcc/numK, label='Train Accuracy')
        ax2.plot(x, CV_ValAcc/numK, label='Validation Accuracy')
        ax2.legend(loc='lower right')
        ax2.set_title('Accuracy')
        f.suptitle('{} fold CV, Dataset ID {}, With DWT: {}'.format(numK, config.datasetID, config.DWT_Input))
        plt.show()
    else:
        print('Running experiment with config.py specifications...')
        Experiment()
