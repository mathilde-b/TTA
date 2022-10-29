import io
import requests
import cv2
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

## Moments returned in xy
def moment_raw(r, i, j):
    b, c, w, h = r.shape
    _x = torch.arange(w)**j
    _y = torch.arange(h)**i
    _XX, _YY = torch.meshgrid(_y, _x)
    gri = torch.einsum("wh,wh->wh",_XX,_YY).type(r.type())
    moment = torch.einsum("bcwh,wh->bc",r,gri)
    return moment

def centroid(M00,M10,M01):
    x_ = torch.div(M10,M00+1e-5)
    y_ = torch.div(M01,M00+1e-5)
    return x_,y_


def get_moments_raw(r):
    M00 = moment_raw(r, 0,0)
    M10 = moment_raw(r, 1,0)
    M01 = moment_raw(r, 0,1)
    M11 = moment_raw(r, 1,1)
    M20 = moment_raw(r, 2,0)
    M02 = moment_raw(r, 0,2)
    M21 = moment_raw(r, 2,1)
    M12 = moment_raw(r, 1,2)
    M30 = moment_raw(r, 3,0)
    M03 = moment_raw(r, 0,3)
    x_, y_ = centroid(M00, M10, M01)
    return M00,M01,M10,M11,M12,M21,M20,M02,M30,M03,x_,y_


def central_moments(r):
    M00, M01, M10, M11, M12, M21, M20, M02, M30, M03, x_, y_ = get_moments_raw(r)
    mu00 = M00
    mu01 = 0
    mu10 = 0
    mu11 = M11 - x_* M01 # = M11 - y_* M10
    mu20 = M20 - x_ * M10
    mu02 = M02 - y_ * M01
    mu21 = M21 - 2*x_ * M11 - y_ * M20 + 2 * x_**2 * M01
    mu12 = M12 - 2*y_ * M11 - x_ * M02 + 2 * y_**2 * M10
    mu30 = M30 - 3*x_ * M20 + 2 * x_**2 * M10
    mu03 = M03 - 3*y_ * M02 + 2 * y_**2 * M01
    return mu00, mu10, mu01, mu11, mu20, mu02, mu21, mu12, mu30, mu03


def eccentricity(r):
    mu00, mu10, mu01, mu11, mu20, mu02, mu21, mu12, mu30, mu03 = central_moments(r)

    inertia = (mu20+mu02)
    #print(mu11,mu20,mu02)
    rest = 4*mu11**2+(mu20-mu02)**2
    #print(rest,"rest")
    rest = torch.sqrt(torch.abs(rest))
    lambda_1 = 1/2*(inertia + rest)
    lambda_2 = 1/2*(inertia - rest)
    #print("lambda_2,lambda_1",lambda_2,lambda_1)
    eccentricity = torch.sqrt(torch.abs(1-lambda_2/(lambda_1+1e-10)))

    return eccentricity

def inertia(r):
    mus, nus = scaleinv_moments(r)
    inertia = nus[2][0]+nus[0][2]
    return inertia


def scaleinv_moments(r):
    mu00, mu10, mu01, mu11, mu20, mu02, mu21, mu12, mu30, mu03 = central_moments(r)
    mus = [[mu00, mu01, mu02, mu03], [mu10, mu11, mu12, 1], [mu20, mu21, 1, 1], [mu30, 1, 1, 1]]
    # mu13, ... are set to 1 for convenience - not computed or used
    get_nus = lambda i, j, mus: torch.div(mus[i][j], ((mu00+1e-5) ** (1 + (i + j) / 2)))
    init_nus = [0,0,0,0]
    nus = [init_nus, init_nus, init_nus, init_nus] # just to initialize
    for ij in [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]:
        #print("nu%d%d\t" % (ij[0], ij[1]), get_nus(*ij, mus))
        nus[ij[0]][ij[1]] = get_nus(*ij, mus)
    return mus, nus


