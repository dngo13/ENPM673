#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: diane
"""

# Imports the libraries needed
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read in the video and take first frame
vid = cv2.VideoCapture('Tag1.mp4')
ret, frame = vid.read()

# Grayscale the frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Video', gray)

# Fast Fourier transform of the grayscaled image
fft = np.fft.fft2(gray)
# Shift the center
fft_shift = np.fft.fftshift(fft)
mag_spect = 20*np.log(np.abs(fft_shift))
# Get the height and width of the image
height, width = gray.shape
height_h, width_h = int(height/2), int(width/2)
# High pass filter
fft_shift[height_h-40:height_h+40, width_h-40:width_h+40] = 0
# Inverse fft to shift it back to its original position
inv_fft = np.fft.ifftshift(fft_shift)
im = np.fft.ifft2(inv_fft)
im = np.abs(im)
 
blur = cv2.GaussianBlur(im, (7,7), 0)

# 50, 150 cv2.THRESH_TOZERO
ret_fft, thres_fft  = cv2.threshold(blur, 30, 150, cv2.THRESH_TOZERO)
ret, thres  = cv2.threshold(gray, 70, 150, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
x, y, w, h = cv2.boundingRect(thres)
thres = cv2.cvtColor(thres, cv2.COLOR_BGR2RGB)

bound_box = cv2.rectangle(thres,(x,y),(x+w,y+h),(0,255,0), 3)

# plt.figure(2)
cv2.imwrite('1a.jpg',thres)

plt.figure(dpi=300)
plt.subplot(221), plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(222), plt.imshow(mag_spect, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([])
plt.yticks([])

plt.subplot(223), plt.imshow(thres_fft, cmap='gray')
plt.title('Inverse FFT with Threshold')
plt.xticks([])
plt.yticks([])

plt.subplot(224), plt.imshow(bound_box, cmap='gray')
plt.title('Image with AR Detection')
plt.xticks([])
plt.yticks([])

plt.show()
cv2.waitKey(0) 

vid.release()
# gray.release()
cv2.destroyAllWindows()