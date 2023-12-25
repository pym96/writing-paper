import os
import numpy as np
from scipy import io
from PIL import Image
import torch
import torchvision.transforms as transforms

class MedicalImagesDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):
        super(MedicalImagesDataset, self).__init__()
        self.files = io.loadmat(os.path.join(dataPath, "setid.mat"))
        if sets == 'train':
            self.files = self.files.get('trnid')[0]
        else:
            self.files = self.files.get('tstid')[0]
        self.transform = transform
        self.datapath = dataPath

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        imgname = "image_%05d.jpg" % self.files[idx]  # 根据您的文件名格式进行修改
        img = self.transform(Image.open(os.path.join(self.datapath, "jpg", imgname)))
        return img * 2 - 1  # 修改或删除这个转换，根据您的需求
