# Zhang Yiyang@XAUAT
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR

origin_raw_data_dir = 'C:\\Users\\21945\\PycharmProjects\\Hands-on-EEG\\new_implement\\data' \
                     '\\orgin_raw_data_slid_window_slice_3000'
model_save = 'C:\\Users\\21945\\PycharmProjects\\Hands-on-EEG\\new_implement\\model'
pic_dir = 'C:\\Users\\21945\\PycharmProjects\\Hands-on-EEG\\new_implement\\pic'


