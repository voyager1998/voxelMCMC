from __future__ import print_function
from voxelMCMC import *
import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
if __name__ == '__main__':
    shadowCluster = generateShadowCluster()
    plt.figure()
    plt.imshow(shadowCluster)
    plt.title("shadow Cluster")

    # Test cases:
    print("For all zeros:")
    x = np.zeros((100, 100), int)
    allS_ks, allsizeofSk = getAllSkandSize(x)
    print(freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk))
    print(probabilityShadowCluster(shadowCluster, x, allS_ks, allsizeofSk))
    print(StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk))    


    print("For same image:")
    x = generateShadowCluster()
    allS_ks, allsizeofSk = getAllSkandSize(x)
    print(freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk))
    print(probabilityShadowCluster(shadowCluster, x, allS_ks, allsizeofSk))
    print(StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk))    

    print("For constructed right image:")
    x = generateShadowCluster()
    for i in range(15, 30):
        for j in range(10,60):
            x[i][j] = 4
    for i in range(40, 70):
        for j in range(20,40):
            x[i][j] = 5
    for i in range(60, 90):
        for j in range(70,95):
            x[i][j] = 6
    plt.figure()
    plt.imshow(x)
    plt.title("constructed right image")
    allS_ks, allsizeofSk = getAllSkandSize(x)
    print(freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk))
    print(probabilityShadowCluster(shadowCluster, x, allS_ks, allsizeofSk))
    print(StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk))    

    print("For constructed wrong image:")
    x = generateShadowCluster()
    for i in range(10, 30):
        for j in range(40,60):
            x[i][j] = 3
    for i in range(40, 80):
        for j in range(20,40):
            x[i][j] = 2
    for i in range(60, 90):
        for j in range(60,95):
            x[i][j] = 3
    plt.figure()
    plt.imshow(x)
    plt.title("constructed wrong image")
    allS_ks, allsizeofSk = getAllSkandSize(x)
    print(freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk))
    print(probabilityShadowCluster(shadowCluster, x, allS_ks, allsizeofSk))
    print(StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk))    

    print("For constructed wrong image:")
    x = generateShadowCluster()
    for i in range(10, 40):
        for j in range(10,80):
            x[i][j] = 1
    for i in range(40, 80):
        for j in range(20,40):
            x[i][j] = 2
    for i in range(60, 90):
        for j in range(60,95):
            x[i][j] = 3
    plt.figure()
    plt.imshow(x)
    plt.title("constructed wrong image")
    allS_ks, allsizeofSk = getAllSkandSize(x)
    print(freeSpaceOverlapping(shadowCluster, x, K, allS_ks, allsizeofSk))
    print(probabilityShadowCluster(shadowCluster, x, allS_ks, allsizeofSk))
    print(StrongSensorModel(shadowCluster, x, allS_ks, allsizeofSk))    

    eprint("test")

    plt.show()
