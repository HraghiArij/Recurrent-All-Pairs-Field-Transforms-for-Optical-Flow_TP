import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


import matplotlib.pyplot as plt
DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    print("houni!")
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
 
    return img_flo/255.0

  

    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()
    Result_images=[]
    
    with torch.no_grad():
        image_files=[]
        for image in os.listdir(args.path):
           image_files.append(image)
        images = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        for image_file in images:
    # Process each image file here
             print(image_file)

        #images = glob.glob(os.path.join(args.path, '*.png')) + \
        #         glob.glob(os.path.join(args.path, '*.jpg'))
        
        #images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(os.path.join(args.path,imfile1))
            image2 = load_image(os.path.join(args.path,imfile2))

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            Result_images.append({"image":image1,"flow":flow_up})
          
        
        for idx,image in enumerate(Result_images): 
            img_flo=viz(image['image'], image['flow'])
            
            plt.imsave(os.path.join("C:/Users/Arij/RAFT/Resultat", f"result_{idx}.jpg"),img_flo)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
