import numpy as np
import matplotlib.pyplot as plt

def err_emp(D, h):
    sum = 0
    for item in D:
        # print(item)
        if item[0] <= h and item[1] > 0:
            sum += 1
        elif item[0] > h and item[1] < 0:
            sum += 1

    #err = np.sum(np.logical_and(D[:,0] <= h, D[:, 1] > 0)) \
    #    + np.sum(np.logical_and(D[:,0] > h, D[:, 1] < 0))
    #err /= len(D)
    err = sum / len(D)
    return err

if __name__ == "__main__":
    d_train = np.loadtxt("SY32_P19_TD01_data.csv")

    h_init = 0
    step = 0.001
    min_h = h_init
    min_value = 10000000

    h_array = []
    err_array = []

    while h_init < 10:
        if err_emp(d_train, h_init) == 0:
            min_h = h_init
            min_value = 0
            break
        err =  err_emp(d_train, h_init) 
        print(h_init, err)
        if err < min_value:
            min_h = h_init
            min_value = err

        h_array.append(h_init)
        err_array.append(err)
        h_init += step

    print("Min h: {} - {}".format(min_h, min_value))
    plt.plot(h_array, err_array)
    plt.show()

    # print(len(d_train))



