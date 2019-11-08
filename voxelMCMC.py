from __future__ import print_function
import numpy as np
from scipy.optimize import basinhopping
from scipy.optimize import HessianUpdateStrategy
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import sys

K = 10
C = 3
NEGINF = -99999999999
IMGWH = 10
FREESPACEWEIGHT = 10


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def generateShadowCluster():
    shadowCluster = np.zeros((IMGWH, IMGWH), int)
    for i in range(int(0.1*IMGWH), int(0.3*IMGWH)):
        for j in range(int(0.1*IMGWH), int(0.8*IMGWH)):
            shadowCluster[i][j] = 1
    for i in range(int(0.4*IMGWH), int(0.8*IMGWH)):
        for j in range(int(0.2*IMGWH), int(0.4*IMGWH)):
            shadowCluster[i][j] = 2
    for i in range(int(0.6*IMGWH), int(0.9*IMGWH)):
        for j in range(int(0.6*IMGWH), int(0.95*IMGWH)):
            shadowCluster[i][j] = 3
    return shadowCluster


def getBelObjectMask(x, k):
    result = x == k
    return result


def getAllSkandSize(x):
    allS_ks = [None] * (K+1)
    allsizeofSk = [None] * (K+1)
    for k in range(K+1):
        S_k = getBelObjectMask(x, k)
        allS_ks[k] = S_k
        allsizeofSk[k] = np.count_nonzero(S_k)
    return allS_ks, allsizeofSk


def obj_fspace_overlapping(shadowCluster, x, k, allS_ks, allsizeofSk):  # return the log
    S_k = allS_ks[k]
    sizeofSk = allsizeofSk[k]
    if sizeofSk == 0:
        return 0
    sizeofS_k0 = 0
    for i in range(np.shape(shadowCluster)[0]):
        for j in range(np.shape(shadowCluster)[1]):
            if S_k[i][j] and shadowCluster[i][j] == 0:
                sizeofS_k0 += 1
    if sizeofS_k0 == sizeofSk:
        return NEGINF
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
    if sizeofS_00 == 0:
        return NEGINF
    return sizeofS0 * np.log(sizeofS_00/sizeofS0)


def freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk):
    result = 0
    result += fspace_fspace_overlapping(shadowCluster, x, allS_ks, allsizeofSk)
    for k in range(1, K+1):
        result += obj_fspace_overlapping(shadowCluster,
                                         x, k, allS_ks, allsizeofSk)
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
        if sizeofSkc == 0:
            continue
        entrophy += sizeofSkc/sizeofSk * math.log(sizeofSkc/sizeofSk, C)
    return -entrophy


def probabilityShadowCluster(shadowCluster, x, allS_ks, allsizeofSk):
    p = 0
    for k in range(1, K + 1):
        temp = entrophySk(shadowCluster, x, k, allS_ks, allsizeofSk)
        if temp == 1:
            p += NEGINF
        else:
            p += allsizeofSk[k] * np.log(1 - temp)
    return p


def StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk):
    return FREESPACEWEIGHT * freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk)\
            + probabilityShadowCluster(shadowCluster, x, allS_ks, allsizeofSk)


def StrongSensorModelforMCMC(x):
    x = np.reshape(x, (IMGWH, IMGWH))
    x = x.astype(int)
    shadowCluster = generateShadowCluster()
    allS_ks, allsizeofSk = getAllSkandSize(x)
    return -StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk)


class State(object):
    def __init__(self, width=IMGWH):
        self.x0 = np.ndarray((IMGWH, IMGWH), int)
        for i in range(np.shape(self.x0)[0]):
            for j in range(np.shape(self.x0)[1]):
                self.x0[i][j] = random.randint(0,K)
        self.shadowCluster = generateShadowCluster()


class ChangeRandomly(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        i = random.randint(0, IMGWH-1)  # endpoints included
        j = random.randint(0, IMGWH-1)
        x[i*IMGWH + j] = random.randint(0, K)
        return x


class ChangeToNeighbourLabel(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        i = random.randint(0, IMGWH-1)  # endpoints included
        j = random.randint(0, IMGWH-1)
        choice = random.randint(0,1)
        # choice = 1
        if choice == 0:
            x[i*IMGWH + j] = 0
        else:
            dx = random.randint(-1,1)
            dy = random.randint(-1,1)
            index = int(((i+dy)*IMGWH + j+dx)%(IMGWH**2))
            x[i*IMGWH + j] = int(x[index])
        return x


def progressCallback(x, f, accepted):
    eprint("at minima %.4f accepted %d" % (f, int(accepted)))
    # plt.figure()
    plt.imshow(np.reshape(x, (IMGWH, IMGWH)))
    plt.title("Current guess")
    plt.savefig("Current_guess.png")


if __name__ == '__main__':
    x0 = np.ndarray((IMGWH, IMGWH), int)
    for i in range(np.shape(x0)[0]):
        for j in range(np.shape(x0)[1]):
            x0[i][j] = random.randint(0,K)
    plt.figure()
    plt.imshow(x0)
    plt.title("initial guess")
    plt.savefig("initial_guess.png")
    print(x0)

    shadowCluster = generateShadowCluster()
    plt.figure()
    plt.imshow(shadowCluster)
    plt.title("shadow Cluster")
    plt.savefig("shadow_Cluster.png")
# -------------------------------------------------
    allS_ks, allsizeofSk = getAllSkandSize(x0)

    print(freeSpaceOverlapping(shadowCluster, x0, K, allS_ks, allsizeofSk))
    print(probabilityShadowCluster(shadowCluster, x0, allS_ks, allsizeofSk))
    print(StrongSensorModel(shadowCluster, x0, allS_ks, allsizeofSk))
    print(StrongSensorModelforMCMC(x0))

    takestep = ChangeToNeighbourLabel()
    minimizer_kwargs = {"method": "BFGS"}
    # minimizer_kwargs = {"method": "trust-krylov", "jac": False, "hess": HessianUpdateStrategy}
    mcmc = basinhopping(StrongSensorModelforMCMC, x0, niter=100, \
            minimizer_kwargs=minimizer_kwargs, disp=True,take_step=takestep, \
                stepsize=K, callback=progressCallback)
    print(type(mcmc))
    print(np.shape(mcmc.x))
    print(mcmc.fun)
    print(np.reshape(mcmc.x, (IMGWH, IMGWH)))
    plt.figure()
    plt.imshow(np.reshape(mcmc.x, (IMGWH, IMGWH)))
    plt.title("final guess")
    plt.savefig("final_guess_100.png")

