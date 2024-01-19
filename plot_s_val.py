from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, GTA, GTAV
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import pandas as pd

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
from tensorboardX import SummaryWriter
import pandas as pd
train_transform = et.ExtCompose([
            et.ExtResize(size= (1914,1052) ),
    et.ExtRandomCrop(size=(768,768)),
    et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    et.ExtRandomHorizontalFlip(),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

val_transform = et.ExtCompose([
    et.ExtResize( (768,768)  ),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

train_dst = GTA(root='/media/fahad/Crucial X8/Mohamed/GTA/',
                        split='all', transform=train_transform)
val_dst = Cityscapes(root='/media/fahad/Crucial X8/gta5/gta/',
                        split='val', transform=val_transform)
batch_size=1
gta_loader = data.DataLoader(
    train_dst, batch_size=batch_size, shuffle=True, num_workers=4,
    drop_last=True)  # drop_last=True to ignore single-image batches.
cs_loader = data.DataLoader(
    
    val_dst, batch_size=batch_size, shuffle=True, num_workers=4)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gta_s_vls=[]
cs_s_vls=[]
cpt=0
for i, (images, _) in tqdm(enumerate(gta_loader)):

            
            if cpt<500 :
                    images = images.to(device, dtype=torch.float32)
                    s=torch.linalg.svdvals(images) 
                    
                    gta_s_vls.append(s[0])
            else: 
                    break
            cpt+=1
torch.save(gta_s_vls,'gta_s_vls.pt')
print("done gta ", len(gta_s_vls))            
for i, (images, _) in tqdm(enumerate(cs_loader)):

            images = images.to(device, dtype=torch.float32)
            s=torch.linalg.svdvals(images) 
      
            cs_s_vls.append(s[0])
torch.save(cs_s_vls,'cs_s_vls.pt')
print("done cs ", len(cs_s_vls))    