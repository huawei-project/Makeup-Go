# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-14 14:12:52
@LastEditTime: 2019-09-14 16:44:57
@Update: 
'''
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

## Load imageIn 
imageIn = Image.open('data/beautified (903).jpg')
imageIn = imageIn.resize((46*11, 46*11), Image.BILINEAR)
imageIn = np.array(imageIn)

## Forward
X = torch.tensor(np.transpose(imageIn, (2, 0, 1))[np.newaxis]).float().cuda()
Y = model(X).squeeze(0)

imageOut = pca.generate_img(Y, X)
imageOut = imageOut[0].cpu().numpy().transpose((1, 2, 0))
imageOut[imageOut > 255] = 255; imageOut[imageOut < 0] = 0; imageOut = imageOut.astype(np.uint8)

plt.figure()
plt.subplot(131)
plt.title("Input")
plt.imshow(imageIn )
plt.subplot(132)
plt.title("Error")
plt.imshow(imageIn - imageOut)
plt.subplot(133)
plt.title("Output")
plt.imshow(imageOut)
plt.show()
