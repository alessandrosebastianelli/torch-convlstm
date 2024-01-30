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

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        OBSERVATION_WINDOWN_LENGHT  = self.in_lenght + self.ou_lenght
        IMAGE_SIZE                  = (128,128)
        CLOUD_SIZE_RANGE            = (int(IMAGE_SIZE[0]*20/100),int(IMAGE_SIZE[0]*60/100),int(IMAGE_SIZE[1]*20/100),int(IMAGE_SIZE[1]*60/100))
        CLOUD_STARTING_POINTS_RANGE = (0,int(IMAGE_SIZE[0]*20/100),0,int(IMAGE_SIZE[1]*20/100))
        CLOUD_VELOCITY_RANGE        = (1,int(IMAGE_SIZE[0]*2.5/100),1,int(IMAGE_SIZE[1]*2.5/100))
        CLOUD_DIRECTIONS            = (-1,1,-1,1)

        # Cloud size
        cw = random.randint(CLOUD_SIZE_RANGE[0], CLOUD_SIZE_RANGE[1])
        ch = random.randint(CLOUD_SIZE_RANGE[2], CLOUD_SIZE_RANGE[3])        
        # Cloud Position
        x0  = random.randint(CLOUD_STARTING_POINTS_RANGE[0], CLOUD_STARTING_POINTS_RANGE[1])
        y0  = random.randint(CLOUD_STARTING_POINTS_RANGE[2], CLOUD_STARTING_POINTS_RANGE[3])
        # Cloud velocity
        cvx = random.randint(CLOUD_VELOCITY_RANGE[0], CLOUD_VELOCITY_RANGE[1])
        cvy = random.randint(CLOUD_VELOCITY_RANGE[2], CLOUD_VELOCITY_RANGE[3])    
        # Cloud direction
        dx = random.choice([CLOUD_DIRECTIONS[0], CLOUD_DIRECTIONS[1]])
        dy = random.choice([CLOUD_DIRECTIONS[1], CLOUD_DIRECTIONS[2]])
        # Initializing image and cloud
        img   = np.zeros((OBSERVATION_WINDOWN_LENGHT,)+IMAGE_SIZE)
        cloud = np.zeros(IMAGE_SIZE)
        
        
        x = x0
        y = y0
        for i in range(OBSERVATION_WINDOWN_LENGHT):
            for xx in range(x, x+cw):
                for yy in range(y, y+ch):
                    try:
                        img[i, xx, yy] = 1                        
                    except:
                        pass

            x = x + cvx*dx
            y = y + cvy*dx

        
        img = 2*img - 1

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
        self.train_dataset = CloudSimulator(size = 1000, in_lenght=self.in_lenght, ou_lenght=self.ou_lenght)#, transform=transform)
        self.valid_dataset = CloudSimulator(size = 100,  in_lenght=self.in_lenght, ou_lenght=self.ou_lenght)#, transform=transform)
        self.test_dataset  = CloudSimulator(size = 10,   in_lenght=self.in_lenght, ou_lenght=self.ou_lenght)#, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
