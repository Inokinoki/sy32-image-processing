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

def minimum(d_train):
    min_h = 0
    min_value = np.Inf

    h_array = []
    err_array = []

    for h in np.sort(d_train[:, 0]):
        err =  err_emp(d_train, h) 
        if err < min_value:
            min_h = h
            min_value = err

        h_array.append(h)
        err_array.append(err)

    print("Min h: {} - {}".format(min_h, min_value))
    #plt.plot(h_array, err_array)
    #plt.show()

    return min_h, min_value

def evaluate(d_test, h):
    return err_emp(d_test, h)

if __name__ == "__main__":
    D = np.loadtxt("SY32_P19_TD01_data.csv")

    print(type(D))

    index = np.arange(len(D))
    np.random.shuffle(index)
    
    d_trains = []

    for i in range(0, 5):
        d_trains.append(D[index[int(i * len(D) / 5.0):int((i+1) * len(D) / 5.0)]])
        # d_test = D[index[60:120]]

    for i in range(0, len(d_trains)):
        d_test = d_trains[i]
        d_train = None
        for j in range(0, len(d_trains)):
            if j is not i:
                if d_train is None:
                    d_train = d_trains[j]
                else:
                    d_train = d_train + d_trains[j]
        

        h, err = minimum(d_train)
        r = evaluate(d_test, h)
        print(r)
    # print(len(d_train))



