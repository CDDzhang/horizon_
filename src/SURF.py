# Use SURF algorithm to extract keypoint


import cv2
import numpy as np
import src.LS as ols

def draw_horizonline(kp,ResizeImg):
    w,b = horizon_extra(kp)
    pt1 = (0,int(b))
    pt2 = (640,int(640*w+b))
    cv2.line(ResizeImg,pt1,pt2,(0,0,255),2,4)
    return ResizeImg

def SURF_Pro(img):
# surf_process
    surf = cv2.xfeatures2d.SURF_create(400)
    kp,des = surf.detectAndCompute(img,None)
    img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    # w,b = horizon_extra(kp)
    # pt1 = (0,int(b))
    # pt2 = (640,int(640*w+b))
    # cv2.line(img2,pt1,pt2,(0,0,255),2,4)
    return img2,kp

def horizon_extra(kp):
    pt_x = []
    pt_y = []
    pts = []
    for i in range(len(kp)):
        pt_x.append(kp[i].pt[0])
        pt_y.append(kp[i].pt[1])
        pts.append(kp[i].pt)

    w,b,cost = ols.fit(pt_x,pts)

    # print("w is:",w)
    # print("b is:",b)
    # print("cost = ",cost)
    return w,b



# def compute_cost(w,b,points):
#     total_cost = 0
#     M = len(points)
#     for i in range(M):
#         x = points[i][0]
#         y = points[i][1]
#         total_cost += (y-w*x-b)**2
#     return total_cost/M
#
# def fit(pt_x,points):
#     M = len(points)
#     x_bar = np.mean(pt_x)
#     sum_yx = 0
#     sum_x2 = 0
#     sum_delta = 0
#     for i in range(M):
#
#         x = points[i][0]
#         y = points[i][1]
#         sum_yx += y*(x-x_bar)
#         sum_x2 += x**2
#     w = sum_yx/(sum_x2-M*(x_bar**2))
#
#     for i in  range(M):
#         x = points[i][0]
#         y = points[i][1]
#         sum_delta += (y-w*x)
#     b = sum_delta/M
#
#     cost = compute_cost(w,b,points)
#     return w,b,cost

