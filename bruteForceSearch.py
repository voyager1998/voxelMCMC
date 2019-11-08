from voxelMCMC import *

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

    min_score = StrongSensorModelforMCMC(x0)
    final_state = x0

    found = 0

    for i in range(np.shape(x0)[0]):
        for j in range(np.shape(x0)[1]):
            for l in range(1, K+1):
                new_label = int((x0[i][j] + l)%(K+1))
                temp = x0
                temp[i][j] = new_label
                score = StrongSensorModelforMCMC(temp)
                if score < min_score:
                    min_score = score
                    print(min_score)
                    final_state = temp
                if min_score == 0:
                    found = 1
                    break
            if found == 1:
                break
        if found == 1:
            break

    plt.figure()
    plt.imshow(final_state)
    plt.title("final guess")
    plt.savefig("bruteForceSearch.png")
    print(min_score)
    print(final_state)
        