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
# C = 3
NEGINF = -99999999999
IMGH = 10
IMGW = 20
FREESPACEWEIGHT = 10


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def manualGenerateShadow():
    shadowCluster = np.zeros((IMGH, IMGW), int)
    for i in range(int(0.1*IMGH), int(0.3*IMGH)):
        for j in range(int(0.1*IMGW), int(0.8*IMGW)):
            shadowCluster[i][j] = 1
    for i in range(int(0.4*IMGH), int(0.8*IMGH)):
        for j in range(int(0.2*IMGW), int(0.4*IMGW)):
            shadowCluster[i][j] = 2
    for i in range(int(0.6*IMGH), int(0.9*IMGH)):
        for j in range(int(0.6*IMGW), int(0.95*IMGW)):
            shadowCluster[i][j] = 3
    return shadowCluster


def manualGenerateShapeCompletion():
    shapeCompletion = np.zeros((IMGH, IMGW), int)
    for i in range(int(0.1*IMGH), int(0.5*IMGH)):
        for j in range(int(0.0*IMGW), int(0.1*IMGW)):
            shapeCompletion[i][j] = 1
    for i in range(int(0.2*IMGH), int(0.3*IMGH)):
        for j in range(int(0.5*IMGW), int(0.6*IMGW)):
            shapeCompletion[i][j] = 2
    for i in range(int(0.4*IMGH), int(0.7*IMGH)):
        for j in range(int(0.15*IMGW), int(0.3*IMGW)):
            shapeCompletion[i][j] = 3
    for i in range(int(0.1*IMGH), int(0.3*IMGH)):
        for j in range(int(0.8*IMGW), int(0.95*IMGW)):
            shapeCompletion[i][j] = 4
    for i in range(int(0.3*IMGH), int(0.6*IMGH)):
        for j in range(int(0.65*IMGW), int(0.9*IMGW)):
            shapeCompletion[i][j] = 5
    for i in range(int(0.8*IMGH), int(0.9*IMGH)):
        for j in range(int(0.05*IMGW), int(0.2*IMGW)):
            shapeCompletion[i][j] = 6
    for i in range(int(0.5*IMGH), int(0.8*IMGH)):
        for j in range(int(0.4*IMGW), int(0.5*IMGW)):
            shapeCompletion[i][j] = 7
    return shapeCompletion


def generateShadowCluster(shapeCompletion):
    shadowCluster = np.zeros((IMGH, IMGW), int)
    clusterIndex = 1
    isLastColOccupied = 0
    for j in range(IMGW):
        isColOccupied = 0
        for i in range(IMGH):
            if isColOccupied == 1:
                shadowCluster[i][j] = clusterIndex
            else:
                if shapeCompletion[i][j] != 0:
                    isColOccupied = 1
                    shadowCluster[i][j] = clusterIndex
        if isColOccupied == 0 and isLastColOccupied == 1:
            clusterIndex += 1
        isLastColOccupied = isColOccupied
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
    sizeofFreespace = np.count_nonzero(shadowCluster == 0)
    if sizeofFreespace == 0:
        return 0
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
    return sizeofS0 * np.log(sizeofS_00/sizeofFreespace)


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
    C = np.amax(shadowCluster)
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
    x = np.reshape(x, (IMGH, IMGW))
    x = x.astype(int)
    shadowCluster = manualGenerateShadow() # TODO: change it to depending on shape completion
    allS_ks, allsizeofSk = getAllSkandSize(x)
    return -StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk)


def getAllqi(shapeCompletion):
    allq_is = [None] * (K+1)
    allsizeofqi = [None] * (K+1)
    for k in range(K+1):
        q_i = getBelObjectMask(shapeCompletion, k)
        allq_is[k] = q_i
        allsizeofqi[k] = np.count_nonzero(q_i)
    return allq_is, allsizeofqi


def computePqi(allq_is, allsizeofqi, i, allS_ks, allsizeofSk):
    if allsizeofqi[i] == 0:
        return 1
    P = [None] * (K + 1)  # from 0 to K
    P[0] = 1e-10
    for k in range(1, K + 1):
        intersection = allq_is[i] & allS_ks[k]
        sizeofInter = np.count_nonzero(intersection)
        union = allq_is[i] | allS_ks[k]
        sizeofUnion = np.count_nonzero(union)
        P[k] = sizeofInter / sizeofUnion
    Pmax = max(P)
    return Pmax


def ShapeCompletionOverlap(x, allS_ks, allsizeofSk, allq_is, allsizeofqi):
    logP = 0.0
    for i in range(1, K + 1):
        logP += allsizeofqi[i] * np.log(computePqi(allq_is, allsizeofqi, i, allS_ks, allsizeofSk))
    return logP


def ShapeCompletionOverlapForMCMC(x):
    x = np.reshape(x, (IMGH, IMGW))
    x = x.astype(int)
    shapeCompletion = manualGenerateShapeCompletion()
    allS_ks, allsizeofSk = getAllSkandSize(x)
    allq_is, allsizeofqi = getAllqi(shapeCompletion)
    return -ShapeCompletionOverlap(x, allS_ks, allsizeofSk, allq_is, allsizeofqi)


def SensorModelCombined(x):
    x = np.reshape(x, (IMGH, IMGW))
    x = x.astype(int)
    allS_ks, allsizeofSk = getAllSkandSize(x)
    # shapeCompletion = manualGenerateShapeCompletion()
    global shapeCompletion
    allq_is, allsizeofqi = getAllqi(shapeCompletion)
    SCO = -ShapeCompletionOverlap(x, allS_ks, allsizeofSk, allq_is, allsizeofqi)
    shadowCluster = generateShadowCluster(shapeCompletion)
    SSM = -StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk)
    return SSM + SCO


def actionModel(x0, k, di, dj, theta):
    pos_k = np.where(x0 == k)
    # print("position of object ", k, ": ", pos_k)
    action_k = np.zeros((3,3), float)
    center = int(math.floor(len(pos_k[0])/2))
    T_k = np.array([[1, 0, di],
                    [0, 1, dj],
                    [0, 0, 1]])
    O_k = np.array([[1, 0, -pos_k[0][center]],
                    [0, 1, -pos_k[1][center]],
                    [0, 0, 1]])
    R_k = np.array([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]])
    Back_k = np.array([[1, 0, pos_k[0][center]],
                    [0, 1, pos_k[1][center]],
                    [0, 0, 1]])
    action_k = T_k @ Back_k @ R_k @ O_k
    # print("action on object ", k, ": ", action_k)

    newPosk = np.zeros((len(pos_k[0]), 2), int)
    for i in range(len(pos_k[0])):
        posV = np.zeros((3, ))
        posV[0] = pos_k[0][i]
        posV[1] = pos_k[1][i]
        posV[2] = 1
        newPosV = action_k @ posV
        # print(newPosV)
        # newPosV = newPosV.round().astype(int)
        newPosV = np.floor(newPosV).astype(int)
        newPosk[i][0] = newPosV[0]
        newPosk[i][1] = newPosV[1]

    x1 = np.copy(x0)
    x1[x1 == k] = 0
    for i in range(len(newPosk)):
        x1[newPosk[i][0]][newPosk[i][1]] = k
    return x1


"""
class State(object):
    def __init__(self, width=IMGH):
        self.x0 = np.ndarray((IMGH, IMGH), int)
        for i in range(np.shape(self.x0)[0]):
            for j in range(np.shape(self.x0)[1]):
                self.x0[i][j] = random.randint(0,K)
        self.shadowCluster = manualGenerateShadow()
"""

class ChangeRandomly(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        i = random.randint(0, IMGH-1)  # endpoints included
        j = random.randint(0, IMGW - 1)
        choice = random.randint(0,1)
        # choice = 1
        if choice == 0:
            x[i*IMGW + j] = 0
        else:
            x[i*IMGW + j] = random.randint(0, K)
        return x


class ChangeToNeighbourLabel(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, x):
        i = random.randint(0, IMGH-1)  # endpoints included
        j = random.randint(0, IMGW-1)
        choice = random.randint(0,4)
        # choice = 1
        if choice == 0 or choice == 1:
            if x[i*IMGW + j] == 0:
                for t in range(10):
                    i = random.randint(0, IMGH-1)
                    j = random.randint(0, IMGW-1)
                    if x[i*IMGW + j] != 0:
                        break
            x[i*IMGW + j] = 0
        elif choice == 2 or choice == 3:
            if x[i*IMGW + j] == 0:
                for t in range(10):
                    i = random.randint(0, IMGH-1)
                    j = random.randint(0, IMGW-1)
                    if x[i*IMGW + j] != 0:
                        break
            dx = random.randint(-1,1)
            dy = random.randint(-1,1)
            index = int(((i+dy)*IMGW + j+dx)%(IMGH*IMGW))
            x[index] = int(x[i*IMGW + j])
        else:
            x[i*IMGW + j] = random.randint(0, K)
        return x


def progressCallback(x, f, accepted):
    # eprint("at minima %.4f accepted %d" % (f, int(accepted)))
    # plt.figure()
    plt.imshow(np.reshape(x, (IMGH, IMGW)))
    plt.title("Current guess")
    plt.savefig("Current_guess.png")


if __name__ == '__main__':
    global shapeCompletion
    shapeCompletion = manualGenerateShapeCompletion()
    plt.figure()
    plt.imshow(shapeCompletion)
    plt.title("shape Completion 0")
    plt.savefig("Shape_Completion_0.png")

    shadowCluster = generateShadowCluster(shapeCompletion)
    plt.figure()
    plt.imshow(shadowCluster)
    plt.title("shadow Cluster 0")
    plt.savefig("shadow_Cluster_0.png")

    # initialize belief of state as first shape completion result
    x0 = shapeCompletion
    plt.figure()
    plt.imshow(x0)
    plt.title("First frame initialization")
    plt.savefig("initialization.png")

    x1 = []
    objIndex = 3
    di = [-3, -3, -2]
    dj = [3, 3, 2]
    theta = [math.pi/4, math.pi/2, math.pi/4]
    for i in range(3):
        temp = actionModel(x0, objIndex, di[i], dj[i], theta[i])
        x1.append(temp)
        plt.figure()
        plt.imshow(temp)
        plt.title("Prediction_" + str(i))
        plt.savefig("prediction_" + str(i)+".png")

    shapeCompletion = np.copy(x1[0])
    shapeCompletion[0][7] = 0
    shapeCompletion[3][6] = objIndex
    shapeCompletion[5][8] = 0
    shapeCompletion[6][8] = 0
    shapeCompletion[7][8] = 0
    shapeCompletion[7][9] = 0
    plt.figure()
    plt.imshow(shapeCompletion)
    plt.title("shape Completion 1")
    plt.savefig("Shape_Completion_1.png")

    shadowCluster = generateShadowCluster(shapeCompletion)
    plt.figure()
    plt.imshow(shadowCluster)
    plt.title("shadow Cluster 1")
    plt.savefig("shadow_Cluster_1.png")

    x1AfterMCMC = []
    bel_x1AfterMCMC = []
    for i in range(3):
        print("---------------------- Update particle ", i, "----------------------")
        takestep = ChangeToNeighbourLabel()
        minimizer_kwargs = {"method": "BFGS"}
        # minimizer_kwargs = {"method": "trust-krylov", "jac": False, "hess": HessianUpdateStrategy}
        mcmc = basinhopping(SensorModelCombined, x1[i], niter=100, \
                minimizer_kwargs=minimizer_kwargs, disp=False, take_step=takestep, \
                    stepsize=K, callback=progressCallback)
        finalGuess = np.reshape(mcmc.x, (IMGH, IMGW))
        x1AfterMCMC.append(finalGuess)
        bel_x1AfterMCMC.append(-mcmc.fun)
        print(type(mcmc))
        print(np.shape(mcmc.x))
        print(mcmc.fun)
        print(finalGuess)
        plt.figure()
        plt.imshow(finalGuess)
        plt.title("final guess " + str(i))
        plt.savefig("final_guess_" + str(i) + "_100.png")
    print("Belief of each particle", bel_x1AfterMCMC)


"""
    x0 = np.ndarray((IMGH, IMGW), int)
    # for i in range(np.shape(x0)[0]):
    #     for j in range(np.shape(x0)[1]):
    #         x0[i][j] = random.randint(0,K)
    x0 = np.array( [[0., 0., 0., 1., 0., 0., 8., 0., 0., 0.],
                    [0., 1., 1., 1., 4., 4., 5., 8., 0., 0.],
                    [0., 5., 1., 1., 4., 8., 8., 8., 0., 0.],
                    [0., 5., 0., 5., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 2., 2., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 2., 2., 0., 0., 0., 0., 0., 0.],
                    [0., 2., 3., 3., 0., 0., 6., 0., 0., 0.],
                    [0., 0., 7., 5., 0., 6., 6., 6., 6., 0.],
                    [0., 1., 0., 0., 4., 0., 9., 6., 6., 0.],
                    [6., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    plt.figure()
    plt.imshow(x0)
    plt.title("initial guess")
    plt.savefig("initial_guess.png")
    print(x0)

    shadowCluster = manualGenerateShadow()
    plt.figure()
    plt.imshow(shadowCluster)
    plt.title("shadow Cluster")
    plt.savefig("shadow_Cluster.png")

    shapeCompletion = manualGenerateShapeCompletion()
    plt.figure()
    plt.imshow(shapeCompletion)
    plt.title("shape Completion")
    plt.savefig("Shape_Completion.png")

# -------------------------------------------------
    allS_ks, allsizeofSk = getAllSkandSize(x0)
    allq_is, allsizeofqi = getAllqi(shadowCluster)

    print(freeSpaceOverlapping(shadowCluster, x0, K, allS_ks, allsizeofSk))
    print(probabilityShadowCluster(shadowCluster, x0, allS_ks, allsizeofSk))
    print(StrongSensorModel(shadowCluster, x0, allS_ks, allsizeofSk))
    print(StrongSensorModelforMCMC(x0))
    print(ShapeCompletionOverlapForMCMC(x0))

    takestep = ChangeToNeighbourLabel()
    minimizer_kwargs = {"method": "BFGS"}
    # minimizer_kwargs = {"method": "trust-krylov", "jac": False, "hess": HessianUpdateStrategy}
    mcmc = basinhopping(SensorModelCombined, x0, niter=300, \
            minimizer_kwargs=minimizer_kwargs, disp=True,take_step=takestep, \
                stepsize=K, callback=progressCallback)
    print(type(mcmc))
    print(np.shape(mcmc.x))
    print(mcmc.fun)
    print(np.reshape(mcmc.x, (IMGH, IMGW)))
    plt.figure()
    plt.imshow(np.reshape(mcmc.x, (IMGH, IMGW)))
    plt.title("final guess")
    plt.savefig("final_guess_100.png")
"""
