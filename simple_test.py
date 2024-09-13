"""
# > Script for testing CD-UDepth on USOD10K dataset 
# > The model parameters can be downloaded at: https://pan.baidu.com/s/1Tfqp300I628oDpw00aRwxQ Password: szcf
# > The transmission estimation method is from the open-source code: https://github.com/bilityniu/underwater_dark_chennel
"""
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
# local libs
from model.CD_UDepth import *
from utils.data import *
from utils.utils import *


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_img_folder", type=str, default="./imgs/RGB")
    parser.add_argument("--loc_tranest_folder", type=str, default="./imgs/trans")
    parser.add_argument("--model_path", type=str, default="./saved_model/CD-UDepth_weights.pth")
    parser.add_argument("--delta", type=float, default=0.7)
    args = parser.parse_args()

    # Create output folder if not exist 
    output_folder = './imgs/output'
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)

    # Use cuda 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = TestDataset(root_dir=args.loc_img_folder,
                                tranest_dir = args.loc_tranest_folder,
                                transform=transforms.Compose([ToTestTensor()]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    # Load specific model
    model = CD_UDepth()
    model.load_state_dict(torch.load(args.model_path))

    print("Model loaded: UDepth")

    model = model.to(device=device)
    model.eval()

    # Testing loop
    for i,sample_batched1 in enumerate (test_loader):
        # Prepare data
        img_fn = sample_batched1['file_name'][0]
        rgb = torch.autograd.Variable(sample_batched1['image'].to(device=device))
        imt = torch.autograd.Variable(sample_batched1['trans'].to(device=device))

        _,out=model(rgb,imt,delta=args.delta)
        out = nn.functional.interpolate(out, [480,640], mode='bilinear', align_corners=True)
        
        img=out.detach().cpu().numpy()
        result=img.reshape(480,640)
        result = Image.fromarray((result * 255).astype(np.uint8))

        plt.imsave(os.path.join(output_folder, img_fn), result, cmap='inferno')

    print("Total images: {0}\n".format(len(test_loader)))
