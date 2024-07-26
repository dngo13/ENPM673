#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: diane
"""

# Imports the libraries needed
import numpy as np
import cv2
from scipy.spatial import distance as dist
import copy


# Function to find homography between the two 
def find_homography(template, overlay):
    # print("template", template)
    A = []
    try:
        for idx in range(len(template)):
            x, y = template[idx][0], template[idx][1]
            u, v = overlay[idx][0], overlay[idx][1]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A)
       
        L = Vh[-1, :] / Vh[-1, -1]
        H = L.reshape(3, 3)
        # print("Homography matrix\n", H)
        return H
    except:
        pass
    
    

def get_ar_tag_info(img):
    # # Grayscale the frame
    warp_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Warped", warp)
    ret, warp_thres = cv2.threshold(warp_gs, 170, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Warped Threshold", warp_thres)
    # 50:150,50:150
    # crop = warp_thres[40:125,40:125].copy()
    # x0 = 0
    # y0 = 0
    # x1 = 150
    # y1 = 150
    # crop = warp_thres[y0+50:y1+1, x0+50:x1+1].copy()
    # cv2.imshow("cropped", crop)
    # print(crop.shape)
    # =====================================================
    angle = 0
    pos = 'UNKNOWN'
    # print("Checking outer border")
    # print('='*30)
    # Top Left
    if 255 in warp_thres[0:15, 0:15]:
        print("Top left corner is white")
        angle = str(180)
        pos = 'TL'
        print(("Tag is upside down at {} degrees").format(angle))
        # crop_tag[0:0 + 24, 0:0+24] = (255, 0, 0) # top left red
    # else:
    #     print("Top left corner is not white")
    # crop_tag[0:0 + 25, 0:0+25] = (255, 0, 0) # top left red 
    # cv2.line(img_copy, (0,24),(24,24), (200,0,0), 1)
        print('='*50)
        return angle, pos
    # Top Right
    elif 255 in warp_thres[0:15, 85:99]:
        print("Top right corner is white")
        angle = str(90)
        pos = 'TR'
        print(("Tag is sideways at {} degrees").format(angle))
    # else:
    #     print("Top right corner is not white")
    # crop_tag[0:0 + 25, 75:75+25] = (255, 0, 0) # top right
        print('='*50)
        return angle, pos
    # Bottom Left
    elif 255 in warp_thres[85:99, 0:15]:
        print("Bottom left corner is white")
        angle = str(270)
        pos = 'BL'
        print(("Tag is sideways at {} degrees").format(angle))
    # else:
    #     print("Bottom left corner is not white")
        print('='*50)
    # crop_tag[75:75 + 25, 0:0+25] = (255, 0, 0) # top right
        return angle, pos
    # Bottom right
    elif 255 in warp_thres[85:99, 85:99]:
        print("Bottom right corner is white")
        angle = str(0)
        pos = 'BR'
        print(("Tag is upright at {} degrees").format(angle))
        # cv2.line(new, (75,75),(100,75), (0,200,100), 1)
        # cv2.line(new, (75,75),(75,100), (0,200,100), 1)
    # else:
    #     print("Bottom right corner is not white")
        return angle, pos
   


def get_contours(thres, contours, hierarchy):  
     desired_contours = []
     final_contours = []
     thres = cv2.cvtColor(thres, cv2.COLOR_BGR2RGB)
     corners = np.array([])
     for cnt in hierarchy[0]:
         if cnt[3] == -1:
             desired_contours.append(contours[cnt[2]])
                 
     for cnt in desired_contours:
         perim =  cv2.arcLength(cnt, True)
         apprx = cv2.approxPolyDP(cnt,  0.01 * perim, True) 
         if len(apprx) == 4:
            corners = apprx
            cv2.drawContours(thres, [corners], 0, (0, 255, 100), 2)
            final_contours.append(corners)
            # print(corners)
            # print("="*30)
            break  
     # cv2.drawContours(thres, [apprx], 0, (255, 0, 0), 3)
     # cv2.imshow("threshold", thres)
     return thres, final_contours, corners


def warp_perspective(cnrs, frame_copy, test_gray, H, dim):  
    try:
        H = np.linalg.inv(H)
        # H = H/H[2][2]
        # frame_copy.shape[0], frame_copy.shape[1]
        warp = np.zeros((dim[0], dim[1], 3))
        for i in range(200):
            for j in range(200):
                x, y, z = np.matmul(H, [i, j, 1])
                # print(x, y, z)
                warp[i][j] = frame_copy[int(y/z)][int(x/z)]
        warp = warp.astype('uint8')
        return warp
    except:
        pass
    


def warp_perspective_test(cnrs, frame_copy, test_gray, H, dim):
    warp = copy.deepcopy(frame_copy)
    # cv2.imshow("testudo gray", test_gray)
    try:
        H = np.linalg.inv(H)
        # H = H/H[2][2]
        c_min, r_min = np.min(cnrs, axis = 0)
        c_max, r_max = np.max(cnrs, axis = 0)
        for j in range(int(r_min), int(r_max)):
            for i in range(int(c_min),int(c_max)):
                s_x, s_y, s_z = np.matmul(H, [i, j, 1])
                if ((int(s_y/s_z) in range(dim[0])) and (int(s_x/s_z) in range(dim[1]))):
                        try:
                            warp[j][i] = test_gray[int(s_y/s_z)][int(s_x/s_z)]
                        except:
                            continue
    except:
        pass
    return warp


def sort_coords(corners):
    # print("Corners\n", corners)
    sorted_cd = np.array([])
    try:
        x_crd = corners[np.argsort(corners[:, 0]), :]
        left = x_crd[:2, :]
        right = x_crd[2:, :]
        left = left[np.argsort(left[:,1]), :]
        (top_lft, bot_lft) = left
        D = dist.cdist(top_lft[np.newaxis], right, "euclidean")[0]
        (bot_rght, top_rght) = right[np.argsort(D)[::-1], :]
        sorted_cd = np.array([top_lft, top_rght, bot_rght, bot_lft], dtype="uint8")
    except:
        pass
    # Return the corners in top-left, top-right, bottom-right, and bottom-left order
    return sorted_cd
    
def orient_pos(pos):
    # print(pos)
    if pos == 'BR':
        # print("Reorienting")
        tst_pos = np.float32([[0, 0],         [200, 0], [200, 200], [0, 200]])
        return tst_pos
    elif pos == 'TF':
        tst_pos = np.float32([[200, 0], [200, 200], [0, 200], [0, 0]])
        return tst_pos
    elif pos == 'TL':
        tst_pos = np.float32([[200, 200], [200, 0], [0, 0], [0, 200]])
        return tst_pos
    elif pos == 'TF':
        tst_pos = np.float32([[0, 200], [0, 0], [200, 0], [200, 200]])
        return tst_pos



def projection_mat(H, K):
    h1 = H[:,0]
    h2 = H[:,1]
    # h3 = H[:,2]
    K_inv = K.T
    lmda = 2/(np.linalg.norm(np.matmul(K_inv, h1)) + np.linalg.norm(np.matmul(K_inv, h2)))
    B_h = lmda * np.matmul(K_inv, H)
    det_B = np.linalg.det(B_h)
    if det_B > 0:
        B = B_h
    else:
         B = -1*B_h    
         
    r_1 = B[:,0]
    r_2 = B[:,1]
    trans = B[:,2]*lmda
    # trans = B[:,2]*lmda
    # print("Translation\n", trans)
    # print(trans.shape)
    r_3 = np.cross(r_1, r_2)
    r_3 = r_3/lmda
    r = np.column_stack((r_1, r_2, r_3))
    r_t = np.column_stack((r_1, r_2, r_3, trans))

    proj_mat = np.matmul(K, r_t)
    return proj_mat, r, trans
 

def draw_cube(frame_copy, cnr_pnts):
    cnr_pnts = np.int32(cnr_pnts).reshape(-1, 2)
    # cv2.imshow("cube test", frame_copy)
    print("Points\n", cnr_pnts)
    frame_copy = cv2.drawContours(frame_copy, [cnr_pnts[:4]], -1,(0, 255, 100), 2)
    for i,j in zip(range(4),range(4,8)):
        fm = cv2.line(frame_copy, tuple(cnr_pnts[i]), tuple(cnr_pnts[j]), (100,255,0), 2)
    fm = cv2.drawContours(frame_copy, [cnr_pnts[4:]],-1,(0,0,255), 3)
    return fm



# main================
def main():
    world_cnrs = np.float32([[0, 0], [199, 0], [199, 199], [0, 199]])

    # Read in the video frames
    # vid = cv2.VideoCapture('Tag0.mp4')
    vid = cv2.VideoCapture('Tag1.mp4')
    # vid = cv2.VideoCapture('Tag2.mp4')
    # vid = cv2.VideoCapture('multipleTags.mp4')
    K = np.float32([[1406.08415449821, 0, 0],
              [2.20679787308599, 1417.99930662800, 0],
              [1014.13643417416, 566.347754321696, 1]])
    K = K.T
    print("K matrix\n", K)
    print(K.shape)
    while vid.isOpened():
        ret, frame = vid.read()
        
        if ret == True:
            frame = cv2.resize(frame, (0,0), fx=0.75, fy=0.75)
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # out = cv2.VideoWriter('Output.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))
            # Grayscale the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_copy = frame.copy()
            # cv2.imshow('Gray Video', gray)
            ret, thres = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            thres, final_contours, corners = get_contours(thres, contours, hierarchy)
            axis_points = np.float32([[0, 0, 0], 
                                    [0, 200, 0], 
                                    [200, 200, 0], 
                                    [200, 0, 0] , 
                                    [0, 0, -200], 
                                    [0, 200,-200], 
                                    [200, 200, -200], 
                                    [200, 0, -200]])

            # try:
            if len(final_contours) > 0:
                    for i in range(len(final_contours)):
                        
                        thres, final_contours, corners = get_contours(thres, contours, hierarchy)
                        corners = corners.reshape(-1, 2)
                        cv2.imshow('Corners Detected', thres)
                        src_corners = np.float32(corners)
                    
                        H = find_homography(src_corners, world_cnrs)
                        
                        test_gray = np.zeros((200,200))
                        # WARP THE AR TAG
                        warp = warp_perspective(src_corners, frame_copy, test_gray, H, (200,200))
                        angle, pos = get_ar_tag_info(warp)
                        # print(pos)
                        tst_pos = orient_pos(pos)
                        world_cnrs = tst_pos
                        P, r, t = projection_mat(H, K)
                        # print("Projection Matrix\n", P)
        
                        corners_pts, jacb = cv2.projectPoints(axis_points, r, t, K, np.zeros((1, 4)))
                        
                        fm = draw_cube(frame_copy, corners_pts)
                        cv2.imshow("Cubes", fm)
                
            # except:
            #     pass
                # out.write(test_frame)
            if not ret:
                break
            # cv2.imshow('Tag', corners)
            # print(warp)
            # 25
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    vid.release()
    # final.release()
    # out.release()
    cv2.destroyAllWindows()
    

    
    
# ================================================    
if __name__ == '__main__':
    main()