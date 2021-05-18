# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:41:18 2021

@author: diane
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def plot_vector(ax, flow, spacing, margin=0, **kwargs):
     """Plots vector field from optical flow
    
     Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    
     # Create dimensions of the plot from the flow
     h, w, *_ = flow.shape
    
     nx = int((w - 2 * margin) / spacing)
     ny = int((h - 2 * margin) / spacing)
     # Set x y limits
     x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
     y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)
    
     flow = flow[np.ix_(y, x)]
     u = flow[:, :, 0]
     v = flow[:, :, 1]
     # Get the arrguments for the vector field
     kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
     ax.quiver(x, y, u, v, **kwargs)
    
     ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
     ax.set_aspect("equal")
    
def dense_optical_flow(method, params=[], to_gray=False):
    print("Calculating...please wait")
    # read the video
    cap = cv2.VideoCapture('Cars On Highway.mp4')
    # Read the first frame
    ret, old_frame = cap.read()
    try:
        old_frame = cv2.resize(old_frame, (0,0), fx=0.5, fy=0.5)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('prob2_cars.mp4', fourcc, 60, (old_frame.shape[0], old_frame.shape[1]))
         # crate HSV & make Value a constant
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255
    
        # Preprocessing for exact method
        if to_gray:
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    except:
        pass
   

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        
        if not ret:
            break
        new_frame = cv2.resize(new_frame, (0,0), fx=0.5, fy=0.5)
        frame_copy = new_frame
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        # plt.figure(dpi=500)
        fig, ax = plt.subplots(figsize=(15,10))
        plot_vector(ax, flow, spacing=25, scale=1, color="#2baeec")
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame
    cap.release()
    # out.release()
    cv2.destroyAllWindows() 
        
def lucas_kanade():
    vid = cv2.VideoCapture('Cars On Highway.mp4')
    # Corner detection parameters
    corners_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    #LK optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),)
    # random colors
    color = np.random.randint(0, 255, (100,3))
    ret, old_frame = vid.read()
    old_frame = cv2.resize(old_frame, (0,0), fx=0.75, fy=0.75)
    gray_old = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    cn_0 = cv2.goodFeaturesToTrack(gray_old, mask=None, **corners_params)
    # crreate mask image to draw on
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = vid.read()
        frame = cv2.resize(frame, (0,0), fx=0.75, fy=0.75)
        # cv2.imshow("Frame", frame)
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # calculate optical flow
        cn_1, st, err = cv2.calcOpticalFlowPyrLK(
            gray_old, gray_frame, cn_0, None, **lk_params
        )
        # Select good points
        good_new = cn_1[st == 1]
        good_old = cn_0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow("Optical flow", img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27: # escape key to break
            break
        if k == ord("c"):
            mask = np.zeros_like(old_frame)
        # Now update the previous frame and previous points
        gray_old = gray_frame.copy()
        cn_0 = good_new.reshape(-1, 1, 2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows() 

    
    
if __name__ == '__main__':
    # lucas_k()
    usr = int(input(("Lucas Kanade or Dense Optical Flow? (1 or 2):  ")))
    if usr == 1:
        lucas_kanade()
    elif usr == 2:
         method = cv2.optflow.calcOpticalFlowSparseToDense
         params = [0.5, 3, 15, 3, 5, 1.2, 0]
         frames = dense_optical_flow(method, to_gray=True)
