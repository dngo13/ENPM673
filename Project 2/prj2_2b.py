#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 05:49:03 2021

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


vid = []
img_list = []
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('prob2p2.mp4', fourcc, 30, (1392, 512), isColor=True)

# Camera calibration matrix
K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
              [0.000000e+00, 9.019653e+02, 2.242509e+02],
              [0.000000e+00, 0.000000e+00, 1.000000e+00]])
# Distortion coefficient matrix
dist = np.array([3.639558e-01, 1.788651e-01, 6.029694e-04,
                 -3.922424e-04, -5.382460e-02])
video = cv2.VideoCapture('challenge_video.mp4')
while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        # cv2.imshow("Video", frame)
        img = frame.copy()
        
        # Gaussian Blur
        blur = cv2.GaussianBlur(frame, (5,5), 3)
        HSV_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        HSV_vect = HSV_img[:, :, 2]
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(13, 13))
        # Apply the histogram
        cl_a = clahe.apply(HSV_vect)
        # Gamma correction
        cl_a = adjust_gamma(cl_a, gamma=1.2)
        HSV_img[:, :, 2] = cl_a
        # Convert back to RGB
        frame = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2BGR)
        
        undst = cv2.undistort(frame, K, dist, None)
        hsl = cv2.cvtColor(undst, cv2.COLOR_BGR2HLS)
        # Crop image to road
        # hsl = hsl[400:800, 400:950]
        cv2.imshow("undst", hsl)
        
        # Find Region of interest  ROI
        # Create mask of image size
        mask = np.zeros_like(hsl)
        # print(img.shape)
        # Channel color
        ch_cnt = img.shape[2]
        skip_clr = (255,)*ch_cnt
        # Mask surroundings, only road
        polygon = np.array([[(275, 670), (600, 465),
                              (720, 465), (1100, 670)]], dtype=np.int32)
        # polygon = np.array([[(0, 400), (400, 400),
        #                      (350, 400), (400, 0)]], dtype=np.int32)
        # Fill the polygon
        cv2.fillPoly(mask, polygon, skip_clr)
        # Mask pixels are nonzero, add the mask to the frame
        masked = cv2.bitwise_and(hsl, mask)
        cv2.imshow("mask", masked)
        # original 
        src_cnrs = np.float32([[600, 465], [720, 465], [275, 650], [1100, 650]])
        # ([[50, 0], [250, 0], [250, 500], [0, 500]],
        # dest_cnrs = np.float32([[150, 0], [600, 0], [20, 500], [700, 500]])
        dest_cnrs = np.float32([[100, 0], [400, 0], [50, 500], [400, 500]])
        homography_mat = cv2.getPerspectiveTransform(src_cnrs, dest_cnrs)
        # # inv_homography = cv2.getPerspectiveTransform(dest_cnrs, src_cnrs)
        # img.shape[1], img.shape[0]
        warp = cv2.warpPerspective(masked, homography_mat, (500,600))
        cv2.imshow("Warped", warp)
        
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # warp = cv2.filter2D(warp, -1, kernel)
        
        # Mask white and yellow pixels
        mask_white_low = np.array([5, 200, 5])
        mask_white_high = np.array([255, 255, 255])
        mask_white = cv2.inRange(warp, mask_white_low, mask_white_high)
        white_lane = cv2.bitwise_and(warp, warp, mask=mask_white).astype(np.uint8)
        
        mask_ylw_low = np.array([20, 120, 90])
        mask_ylw_high = np.array([45, 200, 255])
        mask_yellow = cv2.inRange(warp, mask_ylw_low, mask_ylw_high)
        yellow_lane = cv2.bitwise_and(warp, warp, mask=mask_yellow).astype(np.uint8)
        
        lanes = cv2.bitwise_or(yellow_lane, white_lane)
        new_lanes = cv2.cvtColor(lanes, cv2.COLOR_HLS2BGR)
    
        # Grayscale image
        gray = cv2.cvtColor(new_lanes, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Gray lanes", gray)
        # Canny for edges
        # edges = cv2.Canny(gray, 35, 80)
        # cv2.imshow("Edge", edges)
       
        # Binary threshold to show the lanes
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", thresh)
        
        # Calculate histogram
        histg = np.sum(warp, axis=0)
        output = np.dstack((warp, warp, warp)*255)
        midpt = np.int(histg.shape[0]/2)
        # Compute the left & right max from histogram midpoint
        lft_max = np.argmax(histg[:midpt])
        rght_max = np.argmax(histg[midpt:]) + midpt
        #Image center
        img_cnt = warp.shape[1]/2
        img_cnt = int(img_cnt)
        
        
        # Turn prediction
        mdl_lane = lft_max + ((rght_max - lft_max)/2)
        if (mdl_lane - img_cnt < 0):
            print("Turning left")
            turn = "Left"
        elif (mdl_lane - img_cnt < 9):
            print("Straightaway")
            turn = "Straight"
        else:
            print("Turning right")
            turn = "Right"
            
        print(lft_max)
        # cv2.imshow("final", frame)

    if not ret:
        break

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

video.release()
# out.release()
cv2.destroyAllWindows()
