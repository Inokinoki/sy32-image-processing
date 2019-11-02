import numpy as np

def indice_gini(C, probability):
    sum = 0
    for i in range(0, len(C)):
        sum += probability[i] * (1 - probability[i])
    return sum

points = [(1,6), (2,3), (3,5), (5,4), (0,1), (4,2), (6,0), (7,7)]
classe = [-1, -1, -1, -1, 1, 1, 1, 1]

if __name__ == "__main__":
    p = np.zeros(shape=[len(points[0])])

    for i in range(0, len(p)):
        p[i] = 1.0 / len(p)
    
    # Poid init
    print(p)

    