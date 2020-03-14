import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time
import sys


import Network as nw



MODEL_DIR = "model/"
BEST_MODELE = "best_model.pt"
run_dir = "run-100000"




model = nw.Autoencoder(num_block=3)

model.load_state_dict(torch.load(run_dir + MODEL_DIR + BEST_MODELE))





