from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
from models.stackhourglass import PSMNet

# PSMNet Model Setup
class Args:
    loadmodel = './trained/pretrained_model_KITTI2015.tar'
    model = 'stackhourglass'
    maxdisp = 192
    no_cuda = True  # Not using CUDA
    seed = 1

args = Args()
args.cuda = False

torch.manual_seed(args.seed)

# Model Initialization
model = PSMNet(args.maxdisp)

# Not using nn.DataParallel due to lack of CUDA
device = torch.device('cpu')
model = model.to(device)

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

if args.loadmodel is not None:
    print('Load pretrained model')
    state_dict = torch.load(args.loadmodel, map_location=device)  # Loading on CPU
    state_dict = remove_module_prefix(state_dict['state_dict'])
    model.load_state_dict(state_dict)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL, imgR):
    model.eval()
    with torch.no_grad():
        disp = model(imgL, imgR)
    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()
    return pred_disp

def preprocess_image(image_path):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])
    image = Image.open(image_path).convert('RGB')
    image = infer_transform(image)
    return image

def compute_depth_map_psmnet(left_image_path, right_image_path):
    imgL = preprocess_image(left_image_path)
    imgR = preprocess_image(right_image_path)

    # pad to width and height to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1] // 16
        top_pad = (times + 1) * 16 - imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        right_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0).to(device)
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0).to(device)

    pred_disp = test(imgL, imgR)

    if top_pad != 0 and right_pad != 0:
        img = pred_disp[top_pad:, :-right_pad]
    elif top_pad == 0 and right_pad != 0:
        img = pred_disp[:, :-right_pad]
    elif top_pad != 0 and right_pad == 0:
        img = pred_disp[top_pad:, :]
    else:
        img = pred_disp

    # Normalizing image to uint8
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img = np.uint8(img)
    
    # Applying color map
    img_colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    
    return img_colored

# Setting Dataset Path
dataset = [
    ("./dataset/00/l.png", "./dataset/00/r.png"),
    ("./dataset/01/l.png", "./dataset/01/r.png"),
    ("./dataset/02/l.png", "./dataset/02/r.png"),
    ("./dataset/03/l.png", "./dataset/03/r.png"),
    ("./dataset/04/l.png", "./dataset/04/r.png"),
    ("./dataset/05/l.png", "./dataset/05/r.png")
]

# PSMNet
for idx, (left_image_path, right_image_path) in enumerate(dataset):
    depth_map_psmnet = compute_depth_map_psmnet(left_image_path, right_image_path)
    cv2.imwrite(f'./result/psmnet/{idx}.png', depth_map_psmnet)
