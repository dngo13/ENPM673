#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:44:05 2021

@author: diane
"""

import numpy as np
import cv2

# Function for gamma correction for the video
def adjust_gamma(img, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i
                      in np.arange(0, 256)])
    lut = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))
    return lut


def main():
    vid = cv2.VideoCapture('Night Drive - 2689.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('prob1.mp4', fourcc, 60, (1920, 1080))
    while vid.isOpened():
        ret, frame = vid.read()

        if ret is True:
            cv2.imshow('frame', frame)
            # print("Blur")
            # Blur the video
            blur = cv2.GaussianBlur(frame, (3, 3), 5)
            # Convert video to HSV
            HSV_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            HSV_vect = HSV_img[:, :, 2]
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(13, 13))
            # Apply the histogram
            cl_a = clahe.apply(HSV_vect)
            # Gamma correction
            cl_a = adjust_gamma(cl_a, gamma=1.4)
            HSV_img[:, :, 2] = cl_a
            # Convert back to RGB
            new_img = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2BGR)
            cv2.imshow("Improved", new_img)
            out.write(new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            break
    out.release()
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Main program")
    main()
