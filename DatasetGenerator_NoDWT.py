import torch
import cv2
from tqdm import tqdm
import config
from tkinter import filedialog as fd
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import pickle as pkl
import time

labels = ['benign', 'malignant']

class BreaKHis_DS_NoDWT(torch.utils.data.Dataset):
    def __init__(self,datasetID=000):
        self.imDims = config.imDims # square
        self.numChannels = config.numChannels
        self.cwd = Path.cwd()
        self.loadDS = False
        if not datasetID == 000:
            self.datasetID = datasetID
            self.imDir = None
            self.loadDS = True
        else:
            self.datasetID = np.random.randint(100, 999)
            config.datasetID = self.datasetID
            self.imDir = fd.askdirectory(initialdir=self.cwd, title='Directory of images for DCNN handoff')
        self.pointMap = {}
        self.stackTensor()

    def stackTensor(self):
        if self.loadDS:
            self.data = torch.load(config.tensorDataPath + 'Dataset_{}.pt'.format(self.datasetID))
            with open(config.tensorDataPath + 'Labels_{}.pkl'.format(self.datasetID), 'rb') as infile:
                self.pointMap = pkl.load(infile)
        else:
            imDir_sub = list(Path(self.imDir).glob('*'))

            if len(imDir_sub) != 2 or not all(
                    [x[1] in str(imDir_sub[x[0]]) for x in enumerate(labels)]):
                exit('Check specified imDir, needs just \'imDir/benign/\' and \'imdir/malignant\'')

            pointMap_Idx = 0
            data = torch.empty((0, self.numChannels, self.imDims, self.imDims))
            toTensor = transforms.ToTensor()
            # create a hash map and stack into data tensor

            for folder in imDir_sub:
                points = list(folder.glob('*'))
                print('\nProcessing {}'.format(folder.name))
                time.sleep(.01)
                for point in tqdm(points):

                    if config.NoiseLevelFolder not in list(point.glob('*')):
                        print('{} folder not found for point: {}'.format(config.NoiseLevelFolder, point))
                        pass
                    point = point / config.NoiseLevelFolder

                    if labels[0] in str(folder):
                        self.pointMap[pointMap_Idx] = 0
                    else:
                        self.pointMap[pointMap_Idx] = 1
                    pointMap_Idx += 1
                    DWT_outDir = Path(point)
                    point_sub = list(DWT_outDir.glob('*'))

                    if len(point_sub) != 2:
                        print('Number of files for point: {} not correct!'.format(point))
                        continue

                    im = cv2.imread(str(point_sub[1]))
                    pointTensor = toTensor(im)
                    pointTensor = torch.unsqueeze(pointTensor, 0)
                    data = torch.cat((data, pointTensor))
            self.data = data
            del (data)
            torch.save(self.data, config.tensorDataPath + 'Dataset_{}.pt'.format(self.datasetID))
            with open(config.tensorDataPath + 'Labels_{}.pkl'.format(self.datasetID), 'wb') as outfile:
                pkl.dump(self.pointMap, outfile)
    def __len__(self):
        return len(self.pointMap.keys())

    def __getitem__(self, idx):
        return (self.data[idx], self.pointMap[idx], idx)

    def classKeys(self):
        benign_idx = [i for i in self.pointMap if self.pointMap[i]==0]
        malignant_idx = [i for i in self.pointMap if self.pointMap[i]==1]
        return benign_idx, malignant_idx

# data = BreaKHis_Dataset()
# data.stackTensor()