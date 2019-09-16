# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-14 14:12:52
@LastEditTime: 2019-09-16 09:56:37
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


with open("%s/multidetect.txt" % datapath, 'r') as f:
    detects = eval(f.read())

n = len(detects)
for i, (dirname, (_, [x1, y1, x2, y2], _)) in enumerate(detects.items()):

    print("[%d]/[%d]" % (i + 1, n))

    for j, filename in enumerate(os.listdir("%s/%s" % (datapath, dirname))):

        ## Read imageIn & Crop
        # ---------------------------------------------------
        # inputpath = "%s/%s/%s" % (datapath, dirname, filename)
        # imageIn = Image.open(inputpath)
        # imageIn = np.array(imageIn)

        # # imageIn = cv2.equalizeHist(imageIn)     # hist
        # # imageIn = imageIn + 40; imageIn[imageIn < 0] = 0; imageIn[imageIn > 255] = 255
        
        # imageIn = np.stack([imageIn, imageIn, imageIn], axis=-1)
        # ---------------------------------------------------
        imageCh1 = np.array(Image.open("%s/%s/%s" % (datapath, dirname, "7.jpg" )))
        imageCh2 = np.array(Image.open("%s/%s/%s" % (datapath, dirname, "12.jpg")))
        imageCh3 = np.array(Image.open("%s/%s/%s" % (datapath, dirname, "17.jpg")))
        
        # imageCh1 = cv2.equalizeHist(imageCh1)     # hist
        # imageCh2 = cv2.equalizeHist(imageCh2)     # hist
        # imageCh3 = cv2.equalizeHist(imageCh3)     # hist

        imageIn = np.stack([imageCh1, imageCh2, imageCh3], axis=-1)
        # ---------------------------------------------------

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
        outputfile = "%s/%s/%s" % (outputpath, dirname, filename)
        outputdir  = '/'.join(outputfile.split('/')[:-1])
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        makeupfile = "%s/%s/%s" % (makeuppath, dirname, filename)
        makeupdir  = '/'.join(makeupfile.split('/')[:-1])
        if not os.path.exists(makeupdir):
            os.makedirs(makeupdir)

        cv2.imshow("input", imageIn[:, :, ::-1]); cv2.imshow("output", imageOut[:, :, ::-1]); cv2.imshow("makeup", imageMakeup[:, :, ::-1]); cv2.waitKey(0)

        plt.imsave(outputfile, imageOut)
        plt.imsave(makeupfile, imageMakeup)