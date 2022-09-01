import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import pdb
import numpy as np
import time
import numpy as np
import pytorch_lightning as pl
from CustomDataSet import CocoDataset, Dataset
from decoder import Decoder



class PL_Decoder(pl.LightningModule):
    def __init__(self, trainer):
        super(PL_Decoder, self).__init__()
        self.model = Decoder()
        self.step_now = 0
        self.trainer = trainer

    def forward(self, x):
        return self.model(x)

    def mse_loss(self, input, target):
        r = input[:, 0:1, :, :] - target[:, 0:1, :, :]
        g = (input[:, 1:2, :, :] - target[:, 1:2, :, :])
        b = input[:, 2:3, :, :] - target[:, 2:3, :, :]

        r = torch.mean(r ** 2)
        g = torch.mean(g ** 2)
        b = torch.mean(b ** 2)

        mean = (r + g + b) / 3

        return mean, r, g, b

    def training_step(self, batch, batch_idx):
        if self.current_epoch != self.step_now and self.current_epoch % 10 == 0:
            self.trainer.save_checpoint("epoch-" + str(self.current_epoch) + "_" +
                                        "global_step-" + str(self.global_step) + ".ckpt")
            self.step_now = self.current_epoch

        feats, imgs = batch
        #should convert features with view
        output = self.model(feats)

        return self.mse_loss(output, imgs)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer

class PL_Decoder_data(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

    def setup(self):
        self.train_set = Dataset()
        self.test_set = Dataset()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=10, pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=10, pin_memory=True
        )