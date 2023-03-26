# Path: rl.py
# create an RL agent trained using PPO in PyTorch that takes in 3D states and outputs 3D actions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data as data
import torch.utils.data.distributed as distributed
import torch.utils.data.sampler as sampler
import torch.utils.data.distributed as distributed

import numpy as np
import random
import time
import os
import sys
import argparse
import logging
import math
import copy
import pickle
import json
import datetime
import collections

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.distributions import Independent
from torch.distributions import kl_divergence
from torch.distributions import kl_divergence as kl

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.nn.utils import clip_grad_norm_

class Net(torch.nn.Module):
    def __init__(self):
        # create a network that takes in 3D states and outputs 3D actions
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.tanh(x)
        return x