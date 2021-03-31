# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from tqdm import tqdm

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from torch.nn import functional as F


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from domainbed import datasets
# from domainbed import hparams_registry
# from domainbed import algorithms
# from domainbed.lib import misc
# from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

from tensorboardX import SummaryWriter

import pytorch_lightning as pl

hparams = {}
hparams['lambda'] = 0.1
hparams['d_steps_per_g_step'] = 1
hparams['mlp_width'] = 512
hparams['mlp_depth'] = 2
hparams['pretrained'] = False
hparams['subset'] = True
hparams['upsample'] = False


random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda"

print('STARTED GETTING DATA')
train_joint_dataset = datasets.JointDataset(None, 1, 'train', hparams)
val_joint_dataset = datasets.JointDataset(None, 1, 'val', hparams)

# print(train_dataset)
train_dataset = train_joint_dataset.datasets[0]
val_dataset_1 = val_joint_dataset.datasets[1]
val_dataset_2 = val_joint_dataset.datasets[0]
test_dataset = val_joint_dataset.datasets[1]
print('DONE GETTING DATA OF SHAPES = train {}, val 1 {}, val 2 {}, test{}'.format(
    len(train_dataset), len(val_dataset_1), len(val_dataset_2), len(test_dataset)))

hparams['batch_size'] = 24
hparams['pin_memory'] = True
hparams['weight_decay'] = 1e-6
hparams['workers'] = 4


class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (l1): Linear(...)
      (l2): Linear(...)
    )
    """

    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.network = torchvision.models.resnet50(
            pretrained=hparams['pretrained'])
        self.network.conv1 = torch.nn.Conv1d(
            1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        with torch.no_grad():
            lin_dim = self.network(torch.randn(
                (1, 1, 256, 256))).view(-1).shape[0]
            self.l1 = torch.nn.Linear(lin_dim, 2)
        # self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        # self.print('---------------------------')
        # self.print('---------------------------')
        # self.print('---------------------------')
        # self.print('---------------------------')
        # self.print(list(self.parameters()), 'optimizers configured')
        # self.print('---------------------------')
        # self.print('---------------------------')
        # self.print('---------------------------')
        # self.print('---------------------------')
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        x = torch.relu(self.network(x))
        # print(x.shape)
        x = self.l1(x)
        # x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.long().view(-1)
        loss = F.cross_entropy(y_hat, y.long().view(-1))
        acc = self.accuracy(F.softmax(y_hat, dim=1), y)
        self.log('t_l', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        self.print('----')
        self.print('---- train batch_idx {:5d}, y mean {:.3f}'.format(
            batch_idx, y.float().mean()))
        self.print('---- train batch_idx {:5d}, loss {:.3f}, acc {:.3f}'.format(
            batch_idx, loss.item(), acc.item()))
        self.print('----')
        return loss

    def validation_step(self, batch, batch_idx, val_idx):
        x, y = batch
        y_hat = self(x)
        y = y.long().view(-1)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(F.softmax(y_hat, dim=1), y)
        # self.log('valid_loss', loss)
        self.log('v_l', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('v_a', acc, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)

        self.print('----')
        self.print('---- val {} batch_idx {:5d}, y mean {:.3f}'.format(val_idx,
                                                                       batch_idx, y.float().mean()))
        self.print('---- val {} batch_idx {:5d}, loss {:.3f}, acc {:.3f}'.format(
            val_idx, batch_idx, loss.item(), acc.item()))
        self.print('----')
        return {'loss': loss, 'acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.long().view(-1))
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        # items.pop("v_num", None)
        return items

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitClassifier")
        # parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0005)
        return parent_parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    # parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    # dm = MNISTDataModule.from_argparse_args(args)
    BATCH_SIZE = hparams['batch_size']
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=hparams['workers'], pin_memory=hparams['pin_memory'])
    val_loader_1 = DataLoader(
        val_dataset_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=hparams['workers'], pin_memory=hparams['pin_memory'])
    val_loader_2 = DataLoader(
        val_dataset_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=hparams['workers'], pin_memory=hparams['pin_memory'])
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=hparams['workers'], pin_memory=hparams['pin_memory'])

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version="real",
        name='lightning_logs',
    )

    # ------------
    # model
    # ------------
    model = LitClassifier(args.learning_rate)

    # ------------
    # training
    # ------------
    model_save_callback = ModelCheckpoint(monitor="val_loss", dirpath="/scratch/lhz209/DomainBed/erm_baseline/", filename="ckpt-{epoch:02d}-{val_loss:.2f}", save_top_k=3, mode="min", period=1)
    trainer = pl.Trainer.from_argparse_args(
        args, gpus=-1, logger=logger, limit_train_batches=1.0, limit_val_batches=1.0, precision=16, profiler="simple", callbacks=[model_save_callback])
    trainer.fit(model, train_loader, val_dataloaders=[
                val_loader_1, val_loader_2])

    # ------------
    # testing
    # ------------
    # result = trainer.test(model, test_dataloaders=test_loader, limit_test)
    # print(result)


if __name__ == '__main__':
    # cli_lightning_logo()
    cli_main()
