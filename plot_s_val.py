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


val_transform = et.ExtCompose([
    et.ExtResize( (768,768)  ),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

train_dst = GTA(root='/media/fahad/Crucial X8/Mohamed/GTA/',
                        split='all', transform=val_transform)
val_dst = Cityscapes(root='/media/fahad/Crucial X8/datasets/cityscapes/',
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

            
            if cpt<500:
                    images = images.to(device, dtype=torch.float32)
                    s=torch.linalg.svdvals(images) 
                    print(s.shape)
                    
                    gta_s_vls.append(s[0])
            else: 
                    break
            cpt+=1
print('s0 gta',gta_s_vls[0][:,0])
torch.save(gta_s_vls,'gta_s_vls.pt')
print("done gta ", len(gta_s_vls))  
import matplotlib.pyplot as plt
  
# Plot the singular values
# plt.plot(s[0][0,:].detach().cpu().numpy(), marker='o', linestyle='-', color='r')
# plt.title('Singular Values of the Image')
# plt.xlabel('Singular Value Index')
# plt.ylabel('Singular Value Magnitude (r channel)')
# plt.grid(True)
# plt.savefig('s_vals_magnitude_img0_gta_r.png')
# plt.show()

#
# plt.plot(s[0][1,:].detach().cpu().numpy(), marker='o', linestyle='-', color='g')
# plt.title('Singular Values of the Image')
# plt.xlabel('Singular Value Index')
# plt.ylabel('Singular Value Magnitude (g channel)')
# plt.grid(True)
# plt.savefig('s_vals_magnitude_img0_gta_g.png')
# plt.show()


# plt.plot(s[0][2,:].detach().cpu().numpy(), marker='o', linestyle='-', color='b')
# plt.title('Singular Values of the Image')
# plt.xlabel('Singular Value Index')
# plt.ylabel('Singular Value Magnitude (b channel)')
# plt.grid(True)
# plt.savefig('s_vals_magnitude_img0_gta_b.png')
# plt.show()
for i, (images, _) in tqdm(enumerate(cs_loader)):

            images = images.to(device, dtype=torch.float32)
            s=torch.linalg.svdvals(images) 
      
            cs_s_vls.append(s[0])
torch.save(cs_s_vls,'cs_s_vls.pt')
print("done cs ", len(cs_s_vls))    