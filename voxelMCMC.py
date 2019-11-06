import numpy as np
from scipy.optimize import basinhopping
import random
import matplotlib.pyplot as plt
import math

K = 10
C = 3

def generateShadowCluster():
    shadowCluster = np.ndarray((100,100), int)
    for i in range(10, 30):
        for j in range(10,80):
            shadowCluster[i][j] = 1
    for i in range(40, 80):
        for j in range(20,40):
            shadowCluster[i][j] = 2
    for i in range(60, 90):
        for j in range(60,95):
            shadowCluster[i][j] = 3
    return shadowCluster

def getBelObjectMask(x, k):
    result = x == k
    return result

def getAllSkandSize(x):
    allS_ks = [None] * (K+1)
    allsizeofSk = [None] * (K+1)
    for k in range(K+1):
        S_k = getBelObjectMask(x,k)
        allS_ks[k] = S_k
        allsizeofSk[k] = np.count_nonzero(S_k)
    return allS_ks, allsizeofSk

def obj_fspace_overlapping(shadowCluster, x, k, allS_ks, allsizeofSk): # return the log
    S_k = allS_ks[k]
    sizeofSk = allsizeofSk[k]
    if sizeofSk == 0:
        return 0
    sizeofS_k0 = 0
    for i in range(np.shape(shadowCluster)[0]):
        for j in range(np.shape(shadowCluster)[1]):
            if S_k[i][j] and shadowCluster[i][j] == 0:
                sizeofS_k0 += 1
    return sizeofSk * np.log(1-sizeofS_k0/sizeofSk) 

def fspace_fspace_overlapping(shadowCluster, x, allS_ks, allsizeofSk):
    S_0 = allS_ks[0]
    sizeofS0 = allsizeofSk[0]
    if sizeofS0 == 0:
        return 0
    sizeofS_00 = 0
    for i in range(np.shape(shadowCluster)[0]):
        for j in range(np.shape(shadowCluster)[1]):
            if S_0[i][j] and shadowCluster[i][j] == 0:
                sizeofS_00 += 1
    return sizeofS0 * np.log(sizeofS_00/sizeofS0) 

def freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk):
    result = 0
    result += fspace_fspace_overlapping(shadowCluster, x, allS_ks, allsizeofSk)
    for k in range(1, K+1):
        result += obj_fspace_overlapping(shadowCluster, x, k, allS_ks, allsizeofSk)
    return result

def entrophySk(shadowCluster, x, k, allS_ks, allsizeofSk):
    S_k = allS_ks[k]
    sizeofSk = allsizeofSk[k]
    if sizeofSk == 0:
        return 0
    entrophy = 0
    S_kc = np.multiply(S_k, shadowCluster)
    # plt.figure()
    # plt.imshow(S_kc)
    # plt.title("S k c")
    for c in range(1, C+1):
        sizeofSkc = np.count_nonzero(S_kc == c)
        entrophy += sizeofSkc/sizeofSk * math.log(sizeofSkc/sizeofSk, C)
    return -entrophy

def probabilityShadowCluster(shadowCluster, x, allS_ks, allsizeofSk):
    p = 0
    for k in range(1, K+1):
        p += allsizeofSk[k] * np.log(1- entrophySk(shadowCluster, x, k, allS_ks, allsizeofSk))
    return p

def StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk):
    return freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk) + probabilityShadowCluster(shadowCluster,x,allS_ks,allsizeofSk)

def StrongSensorModelforMCMC(x):
    x = np.reshape(x, (100,100))
    shadowCluster = generateShadowCluster()
    allS_ks, allsizeofSk = getAllSkandSize(x)
    return StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk)

if __name__== '__main__':
    x0 = np.ndarray((100,100), int)
    for i in range(np.shape(x0)[0]):
        for j in range(np.shape(x0)[1]):
            x0[i][j] = random.randint(0,K)
    plt.figure()
    plt.imshow(x0)
    plt.title("initial guess")
    
    shadowCluster = generateShadowCluster()
    plt.figure()
    plt.imshow(shadowCluster)
    plt.title("shadow Cluster")

# -------------------------------------------------
    allS_ks, allsizeofSk = getAllSkandSize(x0)

    print(freeSpaceOverlapping(shadowCluster, x0, K, allS_ks, allsizeofSk))
    print(entrophySk(shadowCluster, x0, 1, allS_ks, allsizeofSk))
    print(StrongSensorModel(shadowCluster, x0, allS_ks, allsizeofSk))

    minimizer_kwargs = {"method": "BFGS"}
    mcmc = basinhopping(StrongSensorModelforMCMC, x0, minimizer_kwargs=minimizer_kwargs)
    plt.figure()
    plt.imshow(mcmc)
    plt.title("final guess")

    plt.show()
