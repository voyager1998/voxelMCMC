from voxelMCMC import *

Kbfs = K
Cbfs = C
IMGWHbfs = IMGH

def arrayPlus1(x, length, maxNum):
    x[length-1] += 1
    for i in range(length, 1, -1):
        if x[i-1] > maxNum:
            x[i-1] = 0
            x[i-2] += 1
    return x

if __name__ == '__main__':
    shadowCluster = np.zeros((IMGWHbfs, IMGWHbfs), int)
    shadowCluster[0][0] = 1
    shadowCluster[0][1] = 1
    shadowCluster[1][0] = 1
    shadowCluster[0][2] = 2
    shadowCluster[1][2] = 2
    shadowCluster[2][2] = 2
    shadowCluster[2][1] = 2

    plt.figure()
    plt.imshow(shadowCluster)
    plt.title("shadow Cluster")
    plt.savefig("shadow_Cluster.png")

    ShapeCompletion = np.zeros((IMGWHbfs, IMGWHbfs), int)
    ShapeCompletion[0][0] = 0
    ShapeCompletion[0][1] = 1
    ShapeCompletion[1][0] = 2
    ShapeCompletion[0][2] = 3
    ShapeCompletion[1][2] = 3
    ShapeCompletion[2][2] = 0
    ShapeCompletion[2][1] = 4

    plt.figure()
    plt.imshow(ShapeCompletion)
    plt.title("Shape Completion")
    plt.savefig("Shape_Completion.png")

    allq_is, allsizeofqi = getAllqi(ShapeCompletion)

    x = np.zeros((IMGWHbfs * IMGWHbfs, ), int)
    print(x)

    fitCount = 0

    while (x[0] <= Kbfs):
        x = np.reshape(x, (IMGWHbfs, IMGWHbfs))
        allS_ks, allsizeofSk = getAllSkandSize(x)
        
        if StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk) == 0 and ShapeCompletionOverlap(x, allS_ks, allsizeofSk, allq_is, allsizeofqi) == 0:
            fitCount += 1
            plt.imshow(x)
            plt.title("solution image")
            plt.savefig("bruteForceResult/solution_"+ str(fitCount) + ".png")

        x = np.reshape(x, (IMGWHbfs * IMGWHbfs, ))
        arrayPlus1(x, IMGWHbfs * IMGWHbfs, Kbfs)
        # print(arrayPlus1(x, IMGWHbfs * IMGWHbfs, Kbfs))