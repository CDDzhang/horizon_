import numpy as np

def compute_cost(w,b,points):
    total_cost = 0
    M = len(points)
    for i in range(M):
        x = points[i][0]
        y = points[i][1]
        total_cost += (y-w*x-b)**2
    return total_cost/M

def fit(pt_x,points):
    M = len(points)
    x_bar = np.mean(pt_x)
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    for i in range(M):
        x = points[i][0]
        y = points[i][1]
        sum_yx += y*(x-x_bar)
        sum_x2 += x**2
    w = sum_yx/(sum_x2-M*(x_bar**2))

    for i in  range(M):
        x = points[i][0]
        y = points[i][1]
        sum_delta += (y-w*x)
    b = sum_delta/M

    cost = compute_cost(w,b,points)
    return w,b,cost



