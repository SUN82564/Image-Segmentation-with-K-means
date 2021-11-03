# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def UNI_Name_kmeans(imgPath,imgFilename,savedImgPath,savedImgFilename,k):

#Write your k-means function here   
#UNI_Name_kmeans(imgPath,imgFilename,savedImgPath,savedImgFilename,k)    
    image = cv2.imread(imgPath+imgFilename)
    image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,image_binary = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)
    
#setting parameters for k-means
    image_reshape = image_color.reshape(-1,3)
    Z = np.float32(image_reshape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
#k-means clustering
    ret,label,center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image_color.shape))
    
# the RGB value for the skin color cluster
    face_color = center[-1]
    
# setting the ranges RGB value for the skin color
    thresh = 45 
    minbgr = np.array([face_color[0]-thresh, face_color[1]-thresh, face_color[2]-thresh])
    maxbgr = np.array([face_color[0]+thresh, face_color[1]+thresh, face_color[2]+thresh])
    
#create a mask
    mask = cv2.inRange(image_color, minbgr, maxbgr)
    
#apply mask: show just faces and the blackbackgroud
    resultBGR = cv2.bitwise_and(image_color, image_color, mask = mask)
# remove noise with the medium filter method
    img_medianfiltered = cv2.medianBlur(resultBGR, ksize=3)
    
#convert the image to binary image with only face and backgroud
    face_gray = cv2.cvtColor(img_medianfiltered, cv2.COLOR_BGR2GRAY)
    _,face_binary = cv2.threshold(face_gray, 127, 255, cv2.THRESH_BINARY)
#remove the noises in the binary image to better locate the face
    face_binary_filtered = cv2.medianBlur(face_binary, ksize = 19)

    
# Find the location by looking for the borders:
    x = []
    y = []
    lmt = 2

    for i in range(0,len(face_binary_filtered)-lmt):
        for j in range(0, len(face_binary_filtered[i])-lmt):
            if (face_binary_filtered[i][j] == 255) and (face_binary_filtered[i+lmt][j]) == 255 and (face_binary_filtered[i][j+lmt]) == 255:
                x.append(i)
                y.append(j)

#Height
    h = np.max(x) - np.min(x)
#width
    w = np.max(y) - np.min(y)
#starting point
    start_y = np.max(x) - h - lmt
    start_x = np.max(y) - w - lmt
#ending point
    end_y = np.max(x) - lmt
    end_x= np.max(y) - lmt

#draw the rectangle on the face
    face_detected = cv2.rectangle(image_color, (start_x,start_y),(end_x,end_y),(0,255,0),2)
    face_detected = cv2.cvtColor(face_detected, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(savedImgPath+savedImgFilename,face_detected)
    
    
    if __name__ == "__main__":
        imgPath='/Users/sunflower/Desktop/GR5293_Image_Analysis/' #Write your own path
        imgFilename="face_d2.jpg"
        savedImgPath=r'/Users/sunflower/Desktop/GR5293_Image_Analysis/' #Write your own path
        savedImgFilename="face_d2_face.jpg"    
        k=4 
        UNI_Name_kmeans(imgPath,imgFilename,savedImgPath,savedImgFilename,k)


print("Short answer: The limitation I found using the K-means method is that \
1. If we don't use the elbow method, we can still recognise the major color groups for a simple picture and choose the k value manually, but that would be difficult when the pictures are more complex. \
2. After clustering and turning the picture into binary format, we can still observe some noises when the background \
has similar colors, which required further process to remove these noices to better locate the position. +\
3. When setting the acceptable ranges of RGB values of the facecolor, it required me to try many times to get the best \
result.")
