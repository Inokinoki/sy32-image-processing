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

def minimum(d_train, begin=0, end=10, step=0.001):
    h_init = begin
    min_h = h_init
    min_value = 10000000

    h_array = []
    err_array = []

    for h in np.sort(d_train[:, 0]):
        err =  err_emp(d_train, h) 
        if err < min_value:
            min_h = h
            min_value = err

        h_array.append(h)
        err_array.append(err)

    """while h_init < end:
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
    """
    #print("Min h: {} - {}".format(min_h, min_value))
    #plt.plot(h_array, err_array)
    #plt.show()

    return min_h

if __name__ == "__main__":
    d_train = np.loadtxt("SY32_P19_TD01_data.csv")

    print(minimum(d_train))

    # print(len(d_train))



