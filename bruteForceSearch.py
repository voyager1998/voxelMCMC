from voxelMCMC import *

Kbfs = 3
Cbfs = 2
IMGWHbfs = 3

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

    x = np.zeros((IMGWHbfs * IMGWHbfs, ), int)
    print(x)

    fitCount = 0

    while (x[0] <= Kbfs):
        x = np.reshape(x, (IMGWHbfs, IMGWHbfs))
        allS_ks = [None] * (Kbfs+1)
        allsizeofSk = [None] * (Kbfs+1)
        for k in range(Kbfs+1):
            S_k = getBelObjectMask(x, k)
            allS_ks[k] = S_k
            allsizeofSk[k] = np.count_nonzero(S_k)
        
        if StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk) == 0:
            fitCount += 1
            plt.imshow(x)
            plt.title("solution image")
            plt.savefig("bruteForceResult/solution_"+ str(fitCount) + ".png")

        x = np.reshape(x, (IMGWHbfs * IMGWHbfs, ))
        arrayPlus1(x, IMGWHbfs * IMGWHbfs, Kbfs)
        # print(arrayPlus1(x, IMGWHbfs * IMGWHbfs, Kbfs))