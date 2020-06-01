import torch
import torch.nn as nn
import numpy as np

import torch
from triplet_sperate import *
from resnet import *
from loss import *
from scipy.spatial.distance import cdist

from scipy.spatial.distance import cdist
x = torch.randn(3,5,requires_grad=True)
y = torch.randn(3,5,requires_grad=True)

z = x + y
print(z)

with torch.no_grad():
    z = x + y
    print(z)