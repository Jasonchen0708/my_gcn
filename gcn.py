import torch
import numpy as np
import matplotlib as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as parameter



class BasicGcn(nn.Module):
    def __init__(self,A, input_szie,out_szie):
        super(BasicGcn, self).__init__()

