from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np
import random
import torch

class CloudSimulator(Dataset):
    def __init__(self, size=3000, in_lenght=96, ou_lenght=48, transform=None):
        self.size      = size
        self.in_lenght = in_lenght
        self.ou_lenght = ou_lenght
        self.transform = transform


        self.OBSERVATION_WINDOWN_LENGHT  = self.in_lenght + self.ou_lenght
        self.IMAGE_SIZE                  = (128,128)
        self.CLOUD_SIZE_RANGE            = (int(self.IMAGE_SIZE[0]*20/100),int(self.IMAGE_SIZE[0]*60/100),int(self.IMAGE_SIZE[1]*20/100),int(self.IMAGE_SIZE[1]*60/100))
        self.CLOUD_STARTING_POINTS_RANGE = (0,int(self.IMAGE_SIZE[0]*20/100),0,int(self.IMAGE_SIZE[1]*20/100))
        self.CLOUD_VELOCITY_RANGE        = (1,int(self.IMAGE_SIZE[0]*2.5/100),1,int(self.IMAGE_SIZE[1]*2.5/100))
        self.CLOUD_DIRECTIONS            = (-1,1,-1,1)


        self.cws = np.arange(self.CLOUD_SIZE_RANGE[0], self.CLOUD_SIZE_RANGE[1])
        self.chs = np.arange(self.CLOUD_SIZE_RANGE[2], self.CLOUD_SIZE_RANGE[3])

        self.xs = np.arange(self.CLOUD_STARTING_POINTS_RANGE[0], self.CLOUD_STARTING_POINTS_RANGE[1])
        self.ys = np.arange(self.CLOUD_STARTING_POINTS_RANGE[2], self.CLOUD_STARTING_POINTS_RANGE[3])

        self.cvxs = np.arange(self.CLOUD_VELOCITY_RANGE[0], self.CLOUD_VELOCITY_RANGE[1])
        self.cvys = np.arange(self.CLOUD_VELOCITY_RANGE[2], self.CLOUD_VELOCITY_RANGE[3])


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Cloud size
        cw = random.choice(self.cws)
        ch = random.choice(self.chs)
        # Cloud Position
        x0 = random.choice(self.xs)
        y0 = random.choice(self.ys)
        # Cloud velocity
        cvx = random.choice(self.cvxs)
        cvy = random.choice(self.cvys)
        # Cloud direction
        dx = random.choice([self.CLOUD_DIRECTIONS[0], self.CLOUD_DIRECTIONS[1]])
        dy = random.choice([self.CLOUD_DIRECTIONS[1], self.CLOUD_DIRECTIONS[2]])
        # Initializing image and cloud
        img   = np.zeros((self.OBSERVATION_WINDOWN_LENGHT,)+self.IMAGE_SIZE)
        cloud = np.zeros(self.IMAGE_SIZE)
        
        x = x0
        y = y0
        for i in range(self.OBSERVATION_WINDOWN_LENGHT):
            for xx in range(x, x+cw):
                for yy in range(y, y+ch):
                    try:
                        img[i, xx, yy] = 1                        
                    except:
                        pass

            x = x + cvx*dx
            y = y + cvy*dx

        
        #img = 2*img - 1

        in_seq = img[:self.in_lenght]
        ou_seq = img[self.in_lenght:]

        in_seq = in_seq[:, None, ...]
        in_seq = torch.tensor(in_seq).float()
        
        ou_seq = ou_seq[None, ...]
        ou_seq = torch.tensor(ou_seq).float()


        return in_seq, ou_seq

class SimulatedCloudsDataset(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=9, in_lenght=96, ou_lenght=48):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_lenght = in_lenght
        self.ou_lenght = ou_lenght

    def setup(self, stage=None):
        #transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = CloudSimulator(size = 10000, in_lenght=self.in_lenght, ou_lenght=self.ou_lenght)#, transform=transform)
        self.valid_dataset = CloudSimulator(size = 100,  in_lenght=self.in_lenght, ou_lenght=self.ou_lenght)#, transform=transform)
        self.test_dataset  = CloudSimulator(size = 10,   in_lenght=self.in_lenght, ou_lenght=self.ou_lenght)#, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
