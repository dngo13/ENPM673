#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 05:49:03 2021

@author: diane
"""

import numpy as np
import cv2
# import glob
# import math

frame_sz = (1392, 512)

vid = []
img_list = []
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# Data set 1 - images to video
# =============================================================================
# for file_name in glob.glob('./data/*.png'):
#     img_list.append(file_name)
#     img_list.sort()
# for i in img_list:
#     img = cv2.imread(i)
#     height, width, layers = img.shape
#     frame_sz = (width, height)
#     vid.append(img)
# out = cv2.VideoWriter('data.mp4', fourcc, 30, frame_sz)
# for i in range(len(vid)):
#     out.write(vid[i])
# =============================================================================
out = cv2.VideoWriter('prob2p1.mp4', fourcc, 30, (1392, 512), isColor=True)

# Camera calibration matrix
K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
              [0.000000e+00, 9.019653e+02, 2.242509e+02],
              [0.000000e+00, 0.000000e+00, 1.000000e+00]])
# Distortion coefficient matrix
dist = np.array([3.639558e-01, 1.788651e-01, 6.029694e-04,
                 -3.922424e-04, -5.382460e-02])
video = cv2.VideoCapture('data.mp4')
while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        # cv2.imshow("Video", frame)
        img = frame.copy()
        # Undistortion
        # img = cv2.undistort(frame, K, dist, None)
        # Crop image to road
        # crop = undst[225:, 0:900]
        # cv2.imshow("undst", undst)
        
        # Find Region of interest  ROI
        # Create mask of image size
        mask = np.zeros_like(frame)
        # print(img.shape)
        # Channel color
        ch_cnt = img.shape[2]
        skip_clr = (255,)*ch_cnt
        # Mask surroundings, only road
        polygon = np.array([[(120, 506), (600, 250),
                             (700, 250), (950, 506)]], dtype=np.int32)

        # Fill the polygon
        cv2.fillPoly(mask, polygon, skip_clr)
        # Mask pixels are nonzero, add the mask to the frame
        masked = cv2.bitwise_and(frame, mask)
        cv2.imshow("mask", masked)
         
        # Grayscale image
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        # Gaussian Blur
        blur = cv2.GaussianBlur(gray, (7,7), 7)
        # Canny for edges
        edges = cv2.Canny(blur, 190, 210)
        # Binary threshold to show the lanes
        ret, thresh = cv2.threshold(blur, 205, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", thresh)
        minLineLength = 5
        maxLineGap = 0
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180,
                                25, np.array([]), minLineLength, maxLineGap)
        
        # Create empty arrays for left and right lines on the image
        lft = []
        rght = []
        try:
            if lines is not None:
                for line in lines:  
                    # Take line and create 4 points
                    x1, y1, x2, y2 = line.reshape(4)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255),5)
                    # Ignore vertical lines and horizontal lines
                    if x1 == x2 or y1 == y2:
                        pass
                    # if y1 <= y2:
                    #     pass
                    else:
                        # Determine slope and intercept
                        slope_m = (y2-y1)/(x2-x1)   
                        int_c = y1- slope_m*x1
                        if slope_m < 0:
                            lft.append((slope_m, int_c))
                        elif slope_m >= 0:
                            rght.append((slope_m, int_c))     
                # out.write(img)  
        except TypeError:
            print("Line type error")
        # Take mean of left and right points
        lft_mean = np.mean(lft, axis=0)
        rght_mean = np.mean(rght, axis=0)
        # Get slope and intercept of left and right
        m_lft, yc_lft = lft_mean
        m_rght, yc_rght = rght_mean
        
        # Left Line
        y1_l = img.shape[0]
        y2_l = 255
        x1_l = int((y1_l - yc_lft) / m_lft)
        x2_l = int((y2_l - yc_lft) / m_lft)
        # cv2.circle(frame,(y1_l, x2_l), 5, (0,255,0), -1)
        
        # Right Line
        y1_r = img.shape[0]
        y2_r = 255
        x1_r = int((y1_r - yc_rght) / m_rght)
        x2_r = int((y2_r - yc_rght) / m_rght)
        # cv2.circle(frame,(y_max, x2_r), 3, (255,255,0), -1)
        # Create overlay to be able to add transparency
        overlay = img.copy()
        # If lines cross then ignore
        if x2_r < x2_l:
            pass
        else:
            # Draw fitted line on left and right ;ames
            cv2.line(overlay, (x1_l, y1_l), (x2_l, y2_l),(255,150, 0), 10)
            cv2.line(overlay, (x1_r, y1_r), (x2_r, y2_r), (255, 150, 0), 10)
            bg = np.array([[(x2_l, y2_l), (x1_l, y1_l), (x1_r, y1_r), (x2_r, y2_r)]], dtype=np.int32)
            # Fill the area between the lines
            cv2.fillPoly(overlay, bg, (100, 255, 0))
            cv2.imshow("Overlay", overlay)
            # Add transparency to overlay
            final = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        cv2.imshow("final", final)
        out.write(final)
        
    if not ret:
        break

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    
video.release()
out.release()
cv2.destroyAllWindows()
