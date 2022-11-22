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

class BreaKHis_DS_DWT(torch.utils.data.Dataset):
    def __init__(self,datasetID=000, dsize=10, saveDataset=False):
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
            data = torch.empty((0, self.numChannels, self.imDims, self.imDims))  # What about color channels...?
            toTensor = transforms.ToTensor()
            # create a hash map and stack into data tensor

            for folder in imDir_sub:
                points = list(folder.glob('*'))
                print('\nProcessing {}'.format(folder.name))
                time.sleep(.01)
                for point in tqdm(points):
                    time.sleep(.01)
                    pointTensor = torch.empty((0, self.imDims, self.imDims))
                    if labels[0] in str(folder):
                        self.pointMap[pointMap_Idx] = 0
                    else:
                        self.pointMap[pointMap_Idx] = 1
                    pointMap_Idx += 1
                    DWT_outDir = Path(point / 'DWT Output')
                    point_sub = list(DWT_outDir.glob('*'))

                    if len(point_sub) != self.numChannels:
                        print('Number of images for point: {} not correct!'.format(point))
                        continue

                    for pic_fileName in point_sub:
                        grayImage = cv2.imread(str(pic_fileName), cv2.IMREAD_GRAYSCALE)
                        im = toTensor(grayImage)
                        try:
                            pointTensor = torch.cat((pointTensor, im))
                        except:
                            print('Check dims on point: {}'.format(point))
                            continue
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

# data = BreaKHis_Dataset()
# data.stackTensor()