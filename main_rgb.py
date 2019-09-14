# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-14 14:12:52
@LastEditTime: 2019-09-14 17:41:08
@Update: 
'''
import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from module import CRN, _pca_

torch.set_default_tensor_type('torch.cuda.FloatTensor')

## Load models
pca   = _pca_.PCA()
model = pickle.load(open("models/final_model.pkl", 'rb')); model = model.cuda()

datapath   = "data/makeuporigin"
outputpath = "data/output"
makeuppath = "data/makeup"


with open("%s/rgbdetect.txt" % datapath, 'r') as f:
    detects = eval(f.read())

n = len(detects)
for i, (filename, (_, [x1, y1, x2, y2], _)) in enumerate(detects.items()):

    print("[%d]/[%d]" % (i + 1, n))

    ## Read imageIn & Crop
    inputpath = "%s/%s.jpg" % (datapath, filename)
    imageIn = Image.open(inputpath)
    imageIn = np.array(imageIn)

    h, w = imageIn.shape[:2]
    if w > h:
        cx = w // 2
        imageIn = imageIn[:, cx - h // 2: cx + h // 2]
    else:
        cy = h // 2
        imageIn = imageIn[cy - w // 2: cy + w // 2, :]
        
    imageIn = cv2.resize(imageIn, (46*11, 46*11))

    ## Forward
    X = torch.tensor(np.transpose(imageIn, (2, 0, 1))[np.newaxis]).float().cuda()
    Y = model(X).squeeze(0)
    imageOut = pca.generate_img(Y, X)
    imageOut = imageOut[0].cpu().numpy().transpose((1, 2, 0))
    imageOut[imageOut > 255] = 255; imageOut[imageOut < 0] = 0; imageOut = imageOut.astype(np.uint8)

    imageMakeup = imageIn - imageOut
    imageMakeup[imageMakeup > 255] = 255; imageMakeup[imageMakeup < 0] = 0; imageMakeup = imageMakeup.astype(np.uint8)

    ## Save
    outputfile = "%s/%s.jpg" % (outputpath, filename)
    outputdir  = '/'.join(outputfile.split('/')[:-1])
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    makeupfile = "%s/%s.jpg" % (makeuppath, filename)
    makeupdir  = '/'.join(makeupfile.split('/')[:-1])
    if not os.path.exists(makeupdir):
        os.makedirs(makeupdir)

    plt.imsave(outputfile, imageOut)
    plt.imsave(makeupfile, imageMakeup)