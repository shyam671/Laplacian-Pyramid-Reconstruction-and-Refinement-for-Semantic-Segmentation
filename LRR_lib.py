from __future__ import print_function
from __future__ import division
import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import scipy.misc as m
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from Dataloder.loader import get_loader, get_data_path
from Dataloder.metrics import runningScore
from LRR_loss import cross_entropy2d
from Dataloder.augmentations import *
from LRR_utils import label_downscale
from PIL import Image
import shutil
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torchvision.transforms import Resize, ToPILImage, ToTensor
import torchvision.transforms as transforms
from LRR_model import LRR32s,LRR

from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid