#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: diane
"""

# Imports the libraries needed
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to draw 8x8 grid on the reference marker
def draw_grid(img, line_color, thickness, step, type_ = cv2.LINE_AA):
    print("Drawing grid on the image")
    x = step
    y = step
    while x <= img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType = type_)
        x+=step
    while y <= img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType = type_)
        y+=step
    return img

# Function to crop the image to display the inner 4x4 Tag
def crop_image(img):
    print("Cropping image to inner 4x4")
    x0 = 0
    y0 = 0
    x1 = 150
    y1 = 150
    crop_tag = img[y0+50:y1+1, x0+50:x1+1].copy()
    return crop_tag


img = cv2.imread('ref_marker.png')
img_copy = img.copy()
step = int(img.shape[0]/8)
img_copy = draw_grid(img_copy, (0, 150, 255), 3, step, type_ = cv2.LINE_AA)
# cv2.imshow('Reference Marker',img)
# cv2.imshow('Grid Marker',img_copy)
crop_nolines = crop_image(img)
crop_tag = crop_image(img_copy)
# cv2.imshow('Cropped Image AR Tag', crop_tag)


# white = 255
gray = cv2.cvtColor(crop_tag, cv2.COLOR_BGR2GRAY)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


corners = cv2.goodFeaturesToTrack(gray_img,10,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(gray_img,(x,y),3,150,-1)
    

cv2.imshow('Corners Detected', gray_img)
ret_, new = cv2.threshold(gray, 163, 255, cv2.THRESH_BINARY)

 # =====================================================
print("Checking outer border")
print('='*30)
# Top Left
if 255 in new[0:24, 0:24]:
    print("Top left corner is white")
    angle = str(180)
    print(("Tag is upside down at {} degrees").format(angle))
    # crop_tag[0:0 + 24, 0:0+24] = (255, 0, 0) # top left red
else:
    print("Top left corner is not white")
# crop_tag[0:0 + 25, 0:0+25] = (255, 0, 0) # top left red 
# cv2.line(img_copy, (0,24),(24,24), (200,0,0), 1)
print('='*50)
# Top Right
if 255 in new[0:24, 75:99]:
    print("Top right corner is white")
    angle = str(90)
    print(("Tag is sideways at {} degrees").format(angle))
    
else:
    print("Top right corner is not white")
# crop_tag[0:0 + 25, 75:75+25] = (255, 0, 0) # top right
print('='*50)

# Bottom Left
if 255 in new[75:99, 0:24]:
    print("Bottom left corner is white")
    angle = str(270)
    print(("Tag is sideways at {} degrees").format(angle))
else:
    print("Bottom left corner is not white")
print('='*50)
# crop_tag[75:75 + 25, 0:0+25] = (255, 0, 0) # top right

# Bottom right
if 255 in new[75:99, 75:99]:
    print("Bottom right corner is white")
    angle = str(0)
    print(("Tag is upright at {} degrees").format(angle))
    # cv2.line(new, (75,75),(100,75), (0,200,100), 1)
    # cv2.line(new, (75,75),(75,100), (0,200,100), 1)
else:
    print("Bottom right corner is not white")

# =====================================================
print('='*50)
print("Checking inner border")
print('='*30)
AR_Tag = []
# Top Left
if 255 in new[25:50, 25:50]:
    print("(Inner) Top left corner is white")
    ar_tag_TL = 1
else:
    print("(Inner) Top left corner is not white")
    ar_tag_TL = 0
print('='*50)

# Top Right
if 255 in new[25:50, 50:75]:
    print("(Inner) Top right corner is white")
    ar_tag_TR = 1
else:
    print("(Inner) Top right corner is not white")
    ar_tag_TR = 0
print('='*50)

# Bottom Left
if 255 in new[50:75, 25:50]:
    print("(Inner) Bottom left corner is white")
    ar_tag_BL = 1
else:
    print("(Inner) Bottom left corner is not white")
    ar_tag_BL = 0
print('='*50)


# Bottom right
if 255 in new[50:75, 50:75]:
    print("(Inner) Bottom right corner is white")
    ar_tag_BR = 1
else:
    print("(Inner) Bottom right corner is not white")
    ar_tag_BR = 0
print('='*50)  

AR_Tag.extend([ar_tag_TL, ar_tag_TR, ar_tag_BR, ar_tag_BL])
print("AR Tag ID is", AR_Tag)


plt.figure(dpi=300)
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Reference Marker Image')
plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(img_copy, cmap='gray')
plt.title('Marker with Grid')
plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(crop_tag, cmap='gray')
plt.title('Cropped Image 4x4 Center')
plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(new, cmap='gray')
plt.imshow(new, cmap='gray')
plt.title('Threshold Binary Image')
plt.xticks([]), plt.yticks([])
plt.show()


cv2.waitKey(0) 
cv2.destroyAllWindows()