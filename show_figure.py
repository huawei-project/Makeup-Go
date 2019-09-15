# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-15 09:19:01
@LastEditTime: 2019-09-15 09:47:28
@Update: 
'''
import os
import cv2
import argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def showFigure(origin, makeup, output):
    """
    Params:
        origin, makeup, output: {ndarray(H, W, 3)}
    """
    plt.figure()
    plt.subplot(131)
    plt.title("origin")
    plt.imshow(origin)
    plt.subplot(132)
    plt.title("makeup")
    plt.imshow(makeup)
    plt.subplot(133)
    plt.title("output")
    plt.imshow(output)
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',    '-t',    choices=['rgb', 'multi'], required=True)
    parser.add_argument('--subject', '-sub',  choices=[1, 2, 3, 4, 5], required=True, type=int)
    parser.add_argument('--session', '-sess', default=1, type=int)
    parser.add_argument('--channel', '-c',    default=1, type=int)
    args = parser.parse_args()
    
    originDirName = "data/makeuporigin/{:s}/{:d}/{:d}".format(args.type, args.subject, args.session)
    makeupDirName = "data/makeup/{:s}/{:d}/{:d}".format(args.type, args.subject, args.session)
    outputDirName = "data/output/{:s}/{:d}/{:d}".format(args.type, args.subject, args.session)
    
    for filename in os.listdir(originDirName):

        originFile = os.path.join(originDirName, filename)
        makeupFile = os.path.join(makeupDirName, filename)
        outputFile = os.path.join(outputDirName, filename)

        if args.type == 'multi':
            originFile = "{:s}/{:d}.jpg".format(originFile, args.channel)
            makeupFile = "{:s}/{:d}.jpg".format(makeupFile, args.channel)
            outputFile = "{:s}/{:d}.jpg".format(outputFile, args.channel)

        originImage = np.array(Image.open(originFile))
        makeupImage = np.array(Image.open(makeupFile))
        outputImage = np.array(Image.open(outputFile))

        h, w = originImage.shape[:2]
        if w > h:
            cx = w // 2
            originImage = originImage[:, cx - h // 2: cx + h // 2]
        else:
            cy = h // 2
            originImage = originImage[cy - w // 2: cy + w // 2, :]
        originImage = cv2.resize(originImage, (46*11, 46*11))

        showFigure(originImage, makeupImage, outputImage)