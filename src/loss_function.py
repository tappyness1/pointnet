import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
import numpy as np
from torch import Tensor

def classification_ce_loss():
    return CrossEntropyLoss()