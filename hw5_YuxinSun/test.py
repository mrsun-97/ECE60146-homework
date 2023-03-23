import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
from torch.utils.data.sampler import SubsetRandomSampler
import torch_directml
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import random
import os

# use directml to run codes on AMD GPU
dml = torch_directml.device()


