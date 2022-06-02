import numpy as np
# import imageio
import imageio.v2 as imageio
import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset

class FGVC_aircraft(Dataset):
    def __init__(self, 
                 input_size: int = 448, 
                 root: str = None, 
                 is_train: bool = True, 
                 data_len: int = None):
        super(FGVC_aircraft, self).__init__()
        assert root is not None
        self.interpolation = transforms.InterpolationMode.BILINEAR
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'data', 'images')
        test_img_path = os.path.join(self.root, 'data', 'images')
        train_label_file = open(os.path.join(self.root, 'data', 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'data', 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img = imageio.imread(self.train_img_label[index][0])
            target = self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size, self.input_size), self.interpolation)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img = imageio.imread(self.test_img_label[index][0])
            target = self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), self.interpolation)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)

class FGVC_aircraft_loader(DataLoader):
    def __init__(self, 
                 root_dir: str = None,
                 batch_size: int = 4, 
                 input_size: int = 448):
        assert root_dir is not None
        assert batch_size > 0
        assert input_size > 0

        self.root_dir = root_dir
        self.input_size = input_size
        self.batch_size = batch_size

    def get_dataloader(self):
        trainset = FGVC_aircraft(input_size=self.input_size, 
                                 root=self.root_dir, 
                                 is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                 batch_size=self.batch_size,
                                                 shuffle=True, num_workers=8, drop_last=False)
        testset = FGVC_aircraft(input_size=self.input_size, root=self.root_dir, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)

        return trainloader, testloader

if __name__ == "__main__":
    root_dir = "/home/ubuntu/tungn197/AirCraft_Cls/fgvc-aircraft-2013b"
    dataloader = FGVC_aircraft_loader(root_dir=root_dir)
    trainloader, testloader = dataloader.get_dataloader()
    for data in trainloader:
        print(data[1])
        break