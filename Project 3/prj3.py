#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:59:45 2021

@author: diane
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# pip install opencv-contrib-python


# Function to get user input and return the data for dataset 1-3
def get_data():
    dir = [name for name in os.listdir(".") if os.path.isdir(name)]

    inp = int(input("Choose data set 1, 2 or 3: "))
    if inp > 3:
        print("Try again")
        inp = int(input("Choose data set 1, 2 or 3"))
    if inp == 1:
        print(dir[0])
        img1 = cv2.imread("Dataset 1/im0.png")
        img2 = cv2.imread("Dataset 1/im1.png")
        img1 = cv2.resize(img1, (0, 0), fx=0.3, fy=0.3)
        # cv2.imshow("img1", img1)
        img2 = cv2.resize(img2, (0, 0), fx=0.3, fy=0.3)
        # data1 = np.concatenate((img1, img2), axis=1)
        data = {'cam0': np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]]),
                'cam1': np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]]),
                'doffs': 174.186,
                'baseline': 177.288,
                'width': 2988,
                'height': 2008,
                'ndisp': 180,
                'isint': 0,
                'vmin': 54,
                'vmax': 147,
                'dyavg': 0,
                'dymax': 0}
        # print(data['cam0'])
        # print(data['height'])
        return img1, img2, data

    if inp == 2:
        print(dir[1])
        img1 = cv2.imread("Dataset 2/im0.png")
        img2 = cv2.imread("Dataset 2/im1.png")
        img1 = cv2.resize(img1, (0, 0), fx=0.3, fy=0.3)
        img2 = cv2.resize(img2, (0, 0), fx=0.3, fy=0.3)
        # data2 = np.concatenate((img1, img2), axis=1)
        data = {'cam0': np.array([[4396.869, 0, 1353.072], [0, 4396.869, 989.702], [0, 0, 1]]),
                'cam1': np.array([[4396.869, 0, 1538.86], [0, 4396.869, 989.702], [0, 0, 1]]),
                'doffs': 185.788,
                'baseline': 144.049,
                'width': 2880,
                'height': 1980,
                'ndisp': 640,
                'isint': 0,
                'vmin': 17,
                'vmax': 619,
                'dyavg': 0,
                'dymax': 0}
        return img1, img2, data

    if inp == 3:
        print(dir[2])
        img1 = cv2.imread("Dataset 3/im0.png")
        img2 = cv2.imread("Dataset 3/im1.png")
        img1 = cv2.resize(img1, (0, 0), fx=0.3, fy=0.3)
        img2 = cv2.resize(img2, (0, 0), fx=0.3, fy=0.3)
        data = {'cam0': np.array([[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]]),
                'cam1': np.array([[5806.559, 0, 1543.51], [0, 5806.559, 993.403], [0, 0, 1]]),
                'doffs': 114.291,
                'baseline': 174.019,
                'width': 2960,
                'height': 2016,
                'ndisp': 250,
                'isint': 0,
                'vmin': 38,
                'vmax': 222,
                'dyavg': 0,
                'dymax': 0}
        return img1, img2, data


# Function to apply SIFT and feature matching
def get_points(img1, img2):
    print("--Getting matching features")
    # Create sift
    orb = cv2.ORB_create(nfeatures=1000)
    im1_gry = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2_gry = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(im1_gry, None)
    kp2, des2 = orb.detectAndCompute(im2_gry, None)
    # sift_img = cv2.drawKeypoints(img2, kp2, None)
    # brute force match detection
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    mtch_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:50], None, matchColor=(50, 150, 255),
                               singlePointColor=(255, 50, 0), matchesMask=None, flags=0)
    num = matches[1:30]
    kp1_lst = []
    kp2_lst = []
    rand_eght = []
    # Add keypoints of both to array
    for i in num:
        kp1_lst.append(kp1[i.queryIdx].pt)
        kp2_lst.append(kp2[i.trainIdx].pt)
    kp1_lst = np.asarray(kp1_lst)
    kp2_lst = np.asarray(kp2_lst)
    # Pick 8 random points
    for i in range(8):
        n = random.randint(0, len(kp1_lst)-1)
        #if i not in rand_eght:
        rand_eght.append(n)
    # rand_eght.sort()
    print(rand_eght)  # Index of the random points
    kp1_f = []
    kp2_f = []

    # Append those 8points to final kp list
    for i in rand_eght:
        kp1_f.append((int(kp1_lst[i][0]), int(kp1_lst[i][1])))
        kp2_f.append((int(kp2_lst[i][0]), int(kp2_lst[i][1])))
    # kp1_f.sort()
    # kp2_f.sort()
    kp1_f = np.asarray(kp1_f)
    kp2_f = np.asarray(kp2_f)
    print("KP1\n", kp1_f)
    print("KP2\n", kp2_f)
    # print("List", kp1_lst)
    return mtch_img, kp1_f, kp2_f, kp1_lst, kp2_lst, kp1, kp2, matches


# Calculate fundamental matrix
def find_fundamental(k1, k2):
    A = []
    for i in range(len(k1)):
        # k1 = [u, v]   k1[i][0] = u, k1[i][1] = v
        # k2 = [u', v'] k2[i][0] = u', k2[i][1] = v'
        # A = u1'u1 , u1'v1, u1', v1'u1, v1'v1, v1', u1, v1, 1
        mat = [k2[i][0]*k1[i][0], k2[i][0]*k1[i][1], k2[i][0], k2[i][1]*k1[i][0],
                k2[i][1]*k1[i][1], k2[i][1], k1[i][0], k1[i][1], 1]
        A.append(mat)

    # Calculate SVD from A matrix
    A = np.asarray(A)
    print("A==========\n", A)
    U, S, Vh = np.linalg.svd(A)
    # Least squares method
    L = Vh[-1, :]/Vh[-1, -1]
    # L = Vh[:][-1]
    # print("L\n", L)
    F = L.reshape(3, 3)
# =============================================================================
#     U, S, Vh = np.linalg.svd(F, full_matrices=False)
#     # Set last singular value to 0
#     S[-1] = 0
#     # # Diagonalize
#     S = np.diag(S)
# =============================================================================
    pts1 = np.array(list(k1[0]) + [1])
    pts2 = np.array(list(k2[0]) + [1])
    F_n = np.matmul(np.transpose(pts2), np.matmul(F, pts1))
    # ans_F = abs(F_n)
    # F_n = np.matmul(np.transpose(k2), np.matmul(F, k1))
    # F_n = np.matmul(np.matmul(U,S), Vh)
    # F_n = abs(F_n)
    print("Fundamental matrix\n", F_n)
    # ===== built in
    fund, inliers = cv2.findFundamentalMat(k1, k2, cv2.FM_RANSAC)
    print("In built F\n", fund)
    return fund


# Calculate essential matrix
def find_essential(F, K):
    # E = K^T*F*K
    print("--Calculating essential matrix")
    E = np.matmul((K.T), np.matmul(F, K)) 
    # np.matmul(np.matmul(K.T, F), K)
    # Correct noise by reconstruction with (1, 1, 0)
    U, S, Vh = np.linalg.svd(E)
    S = [1, 1, 0]
    Sn = np.diag(S)
    En = np.matmul(np.matmul(E, Sn), Vh)
    print("Essential matrix\n", En)
    print("="*50)
    return En

# Gets rotational and translational matrices
def get_pose(E, K):
    print("--Getting rotational and translational matrices")
    U, S, Vh = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # t = UWSU^T
    # R = UW^-1V^T
    R = np.matmul(np.matmul(U, np.linalg.inv(W)), Vh)
    T = np.matmul(np.matmul(np.matmul(U, W), S), U.T)
    print("Rotation\n", R)
    print("Translation\n", T)
    return R, T


def rectify(img1, img2, kp1_f, kp2_f, F):
    print("--Rectifying Images")
    # Get image dimensions()
    (Ht1, W1, c1) = img1.shape
    (Ht2, W2, c2) = img2.shape
    # print(img1_n.shape)
    # print(kp1_f)
    print("F", F)
    # Rectify images
    _, H1, H2 = cv2.stereoRectifyUncalibrated(kp1_f, kp2_f, F, imgSize=(W1, Ht1))
    print("H1 Matrix\n", H1)
    print("H2 Matrix\n", H2)
    # Warp Perspepctive
    im1_rect = cv2.warpPerspective(img1, H1, (W1, Ht1))
    im2_rect = cv2.warpPerspective(img2, H2, (W2, Ht2))
    # cv2.imwrite("rectified_1.png", im1_rect)
    # cv2.imwrite("rectified_2.png", im2_rect)
    # cv2.imshow("Rect 1", im1_rect)
    # cv2.imshow("Rect 2", im2_rect)
    return im1_rect, im2_rect, H1, H2


def get_epipolar(img1_r, img2_r, kp1_ls, kp2_ls, kp1, kp2, H1, H2, matches):
    print("--Epipolar lines")
    kp1_nw = []
    kp2_nw = []
    # Transform keypoints to warped perspective frames
    for i in range(len(kp1_ls)):
        X1 = np.array([kp1_ls[i][0], kp1_ls[i][1], 1])
        X2 = np.array([kp2_ls[i][0], kp2_ls[i][1], 1])
        trns_pt1 = np.dot(H1, np.transpose(X1))
        trns_pt2 = np.dot(H2, np.transpose(X2))
        kp1_nw.append((int(trns_pt1[0]/trns_pt1[2]), int(trns_pt1[1]/trns_pt1[2])))
        kp2_nw.append((int(trns_pt2[0]/trns_pt2[2]), int(trns_pt2[1]/trns_pt2[2])))
    
    cnt = 0
    match = matches[1:30]
    # setting up new keypoints according to feature transform
    for m in match:
        # Get the matching keypoints for each image
        kp1[m.queryIdx].pt = kp1_nw[cnt]
        kp2[m.trainIdx].pt = kp2_nw[cnt]
        cnt+=1
    return kp1_nw, kp2_nw
        
# Draw epipolar lines on the images
def drawlines(img1, img2, lines, pts1, pts2):
    row, col, _ = img1.shape
    for row, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(100, 200, 3).tolist())
        x0, y0 = map(int, [0, -row[2]/row[1]])
        x1, y1 = map(int, [col, -(row[2]+row[0]*col)/row[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), (0, 0, 255), 2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        test_img = np.concatenate((img1, img2), axis=1)
        cv2.imshow("test", test_img)
        cv2.imwrite("epipolar_img.png", test_img)
        # plt.scatter(pt1, pt2)
    return img1, img2


# Calculate disparity
def get_disparity(img1, img2, vmin, vmax):
    disp_diff = vmax - vmin
    uniquenessRatio = 5
    speckleWindowSize = 200
    speckleRange = 2
    disp12MaxDiff = 0
    
    stereo = cv2.StereoSGBM_create(minDisparity=vmin, numDisparities=disp_diff,
        blockSize=21,uniquenessRatio=uniquenessRatio,speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,disp12MaxDiff=disp12MaxDiff,
        P1 = 8 * 1 * 21 * 21, P2 = 32 * 1 * 21 * 21)
    disp_sgbm = stereo.compute(img1, img2)
    
    # Normalize the values to a range from 0-255 (Grayscale)
    disp_sgbm = cv2.normalize(disp_sgbm, disp_sgbm, alpha=255,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
    disp_sgbm = np.uint8(disp_sgbm)
    cv2.imshow("Disparity", disp_sgbm)
    cv2.imwrite("disparity_grayscale.png", disp_sgbm)

    return disp_sgbm

# ------MAIN--------
def main():
    print("1 - Calibration")
    # Get data from user input
    img1, img2, data = get_data()
    # Use sift
    mtch_img, kp1_f, kp2_f, kp1_lst, kp2_lst, kp1, kp2, matches = get_points(img1, img2)
    # Calculate Fundamental matrix
    F = find_fundamental(kp1_f, kp2_f)
    # Fn = np.float32([[1.53827837e-07,  3.43014782e-06, -1.21659886e-03],
    #                 [2.57508130e-06,  1.87415583e-06, -1.99711167e-03],
    #                 [-3.80831474e-04, -2.40799466e-03,  1.00000000e+00]])
    # Intrinsic matrix
    K1 = data['cam0']
    K1.reshape(3, 3)
    K2 = data['cam1']
    K2.reshape(3, 3)
    # Calculate essential matrix
    E = find_essential(F, K1)
    R, T = get_pose(E, K1)
    print("2 - Rectification")
    img1_rect, img2_rect, H1, H2 = rectify(img1, img2, kp1_f, kp2_f, F)
    rect_imgs = np.concatenate((img1_rect, img2_rect), axis=1)
    
    kp1_nw, kp2_nw = get_epipolar(img1_rect, img2_rect, kp1_lst, kp2_lst, kp1, kp2, H1, H2, matches)
    kp1_nw = np.array(kp1_nw)
    kp2_nw = np.array(kp2_nw)
    lines1 = cv2.computeCorrespondEpilines(kp1_nw.reshape(-1, 1, 2), 2, F)
    lines2 = cv2.computeCorrespondEpilines(kp2_nw.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img1_rect, img2_rect, lines1, kp1_nw, kp2_nw)
    img5, img6 = drawlines(img2_rect, img1_rect, lines2, kp1_nw, kp2_nw)
    epi = np.concatenate((img3, img5), axis=1)
    
    # Disparity min max from data given
    vmin = data["vmin"]
    vmax = data["vmax"]
    dispar = get_disparity(img1, img2, vmin, vmax)
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(dispar)* 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imshow("Heatmap", heatmap)
    cv2.imwrite("heatmap.png", heatmap)
    # cv2.imshow("Image", img3)
    # cv2.imshow("Image", img5)
    # ========== Show plots ===========
    # Original SIFT image
    plt.figure(dpi=300)
    plt.title('Dataset')
    plt.imshow(mtch_img)
    plt.xticks([]), plt.yticks([])
    plt.savefig("sift.png")
    
    # Rectified images
    plt.figure(dpi=300)
    plt.imshow(rect_imgs)
    plt.title('Rectified')
    plt.xticks([]), plt.yticks([])
    plt.savefig("rectified.png")
    # # Epipolar
    plt.figure(dpi=300)
    # plt.subplot(121),plt.imshow(img5)
    plt.imshow(epi)
    # plt.xticks([]), plt.yticks([])
    # plt.scatter(kp1_nw, kp2_nw)
    plt.title('Epipolar Lines')
    # plt.subplot(122),plt.imshow(img3)
    plt.xticks([]), plt.yticks([])
    plt.savefig("epipolar.png")
    plt.show()
    
    plt.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
