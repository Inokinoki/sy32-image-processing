from sklearn.svm import LinearSVC, SVC
import numpy as np

from sklearn.model_selection import KFold

#def divide(X, Y, n):
#    pass

def find_arg(kernel="rbf"):
    xa = np.loadtxt("phishing_train.data")
    ya = np.loadtxt("phishing_train.label")

    #Cs = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10e1, 10e2, 10e3]
    
    #Cs = np.arange(0.1, 1.0, 0.1)
    #Cs = np.arange(1, 10, 0.5)
    #Cs = np.arange()

    # RBF
    #Cs = np.arange(1, 10, 0.5)
    #Cs = np.arange(4, 8, 0.2)
    Cs = np.arange(6, 7.6, 0.1)

    #Cs = np.arange(2.5, 4.5, 0.1)
    #Cs = np.arange(7.0, 8.0, 0.1)

    #Cs = np.arange(7.0, 7.2, 0.01)

    taux_erreur_totals = []
    for C in Cs:
        # clf = LinearSVC(C=C)
        clf = SVC(C=C, kernel=kernel, gamma="auto")
        kf = KFold(n_splits=5, shuffle=True)

        taux_erreurs = []
        for train_index, valide_index in kf.split(xa):
            x_train, x_valide = xa[train_index], xa[valide_index]
            y_train, y_valide = ya[train_index], ya[valide_index]

            # clf = LinearSVC(C)
            clf.fit(x_train, y_train)

            y_predict = clf.predict(x_valide)
            
            taux_erreur = (np.sum(y_predict!= y_valide))/len(y_predict)
            taux_erreurs.append(taux_erreur)
            # print("Taux d'erreur {}".format(taux_erreur))
        taux_erreur_total = np.mean(taux_erreurs)
        taux_erreur_totals.append(taux_erreur_total)

    for c, e in zip(Cs, taux_erreur_totals):
        print(c, e)

    import matplotlib.pyplot as plt

    plt.plot(Cs, taux_erreur_totals)
    plt.show()

def test(C=1.0, kernel="rbf"):
    xa = np.loadtxt("phishing_train.data")
    ya = np.loadtxt("phishing_train.label")

    # clf = LinearSVC(C=7.09)
    clf = SVC(C=C, kernel=kernel)
    x_test = np.loadtxt("phishing_test.data")
    clf.fit(xa, ya)

    y_predit = clf.predict(xa)

    # error_count = 0
    #for p, l in zip(y_predit, ya):
    #    if p != l:
    #        error_count += 1 
    #    # print(p, l)
    # taux_erreur = error_count / len(y_predit)

    taux_erreur = (np.sum(y_predit!=ya))/len(y_predit)

    print("Taux d'erreur {}".format(taux_erreur))

    y_test = clf.predict(x_test)

    np.savetxt("phishing_result.data.txt", y_test, "%d")

# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
#find_arg('poly')
#print("=========")
#find_arg('linear')
#print("=========")
#find_arg('rbf')
#print("=========")
#find_arg('sigmoid')
#print("=========")
#find_arg('precomputed')

test(6.6, "rbf")

