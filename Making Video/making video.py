#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:35:23 2022

@author: elhamebr
"""

import cv2
import os

image_folder = '1'
video_name = 'video4.avi'


images = []
for i in range(498):
    images.append('%i.png'%i)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    print(image)

cv2.destroyAllWindows()
video.release()