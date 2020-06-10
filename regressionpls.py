#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import  r2_score, mean_squared_error
import pre_traitement
import fonctionnalite
from time import time
import sys

tls = fonctionnalite.tls
listeDef = fonctionnalite.listeDef
listeFeuille = fonctionnalite.listeFeuille
minDas = fonctionnalite.minDas


def getDatatoDataframe(data):
    """
    fonction qui permet de transformer les données en Dataframe
    :param data: données à transformer
    :return: data frame des données
    """
    liste = []
    X = pd.DataFrame(data).values
    for i in X:
        liste.append((i.tolist()))
    return pd.DataFrame(liste)


def prediction(X_train, Y_train, X_test, Y_test):
    """
    fonction qui permet d'apprendre les données d'entrainementa au classifier regression pls via une optimisation par
    cross validation (GridSearchCV)
    :param X_train: données de reflectances d'entraînement
    :param Y_train: classes associées aux reflectances d'entrainement
    :param X_test: données de reflectances de test
    :param Y_test: classes associées aux reflectances de test, si split=False alors Y_test=None
    :return: le classifier optimisé par cross validation
    """
    params = [{'n_components': [i for i in range(1, 40+1)]}]
    #msemin = convergenceNbComposante(X_train, Y_train, X_test, Y_test, plot)
    pls = PLSRegression()
    inner_cv = KFold(n_splits=5, shuffle=True)
    t0 = time()
    grid = GridSearchCV(estimator=pls, param_grid=params, scoring='r2', cv=inner_cv, return_train_score=True)
    print("Calcul Cross Validation de 1 à 40 composantes pour regression PLS")
    grille = grid.fit(X_train, Y_train)
    print("Cross validation optimisation en %0.3fs" % (time() - t0))

    #décommenter si on veut afficher chez résultats de la cross validation sur toutes les compostantes
    #print(pd.DataFrame.from_dict(grille.cv_results_).loc[:, ["params", "mean_test_score"]])
    print("Meilleur parametre : "+str(grille.best_params_))
    print("Meilleur score R2: "+str(grille.best_score_))
    print()
    if Y_test is not None:
        print("Score pour les données de test via split")
        Y_pred = grille.predict(X_test)
        score_p = r2_score(Y_test, Y_pred)
        mse_p = mean_squared_error(Y_test, Y_pred)
        print('R2: %5.3f' % score_p)
        print('MSE: %5.3f' % mse_p)

    return grille


def prepareTraining(X_train,Y_train,wl,savgol=False, msc=False, rdp=False,  justbymoda=True, plot=False):
    """
    fonction qui permet de préparer les données d'entrainement pour la regression pls. En utilisant ou non
    un pré-traitement, Binarise les classes en fonction du choix fait.
    :param X_train: liste des reflectances associées à chaque échantillon
    :param Y_train: liste de classes associées à chaque échantillon
    :param wl: longueurs d'ondes
    :param savgol: booléen si on utilise le pré-traitement Savitzky Golay
    :param msc: booléen si on utilise le pré-traitement MSC
    :param rdp: booléen si on utilise le pré-traitement RDP
    :param justbymoda: booléen on sépare les classes par moda, ou par moda,feuille,das
    :param plot: booléen si on veut un affichage ou non
    :return: une nouvelle liste X de reflectances, une nouvelle liste Y de classes (binaire), la référence ref des
    données passées par MSC (None si msc=False), les longueurs d'ondes wlrdp passées par RDP (=wl si rdp=False)
    """

    #print("prepareTraining: "+str(len(X_train))+" "+str(len(X_train[0])))

    X_train = X_train.astype('float32')
    if savgol is True:
        X_train = pre_traitement.lissageSavitzky(X_train, wl)
    ref = None
    if msc is True:
        X_train, ref = pre_traitement.compute_msc(X_train,wl, plot=plot)
    wlrdp = wl
    if rdp is True:
        wlrdp, X_train = pre_traitement.computeRdpForData(wl, X_train, Y_train, justbymoda, plot=plot)

    X = getDatatoDataframe(X_train)

    #binarisation des classes
    lb = preprocessing.MultiLabelBinarizer()
    if justbymoda is True:
        lb = preprocessing.LabelBinarizer()
    Y = lb.fit_transform(Y_train)
    oneShotDictionary = {}
    for i in range(0,Y.shape[0]):
#        print("Binarized labels training: "+str(Y[i])+" - "+str(Y_train[i])+" -> "+str(str(Y[i]) in oneShotDictionary))
        if (str(Y[i]) in oneShotDictionary) is False:
            oneShotDictionary[str(Y[i])] = Y_train[i]
    return oneShotDictionary, X,Y, ref, wlrdp


def prepareTesting(X_test,Y_test,wl,split, wlrdp,savgol=False,ref=None, rdp=False,  justbymoda=True, plot=False):
    """

    :param X_test: liste des reflectances associées à chaque échantillon de test
    :param Y_test: liste de classes associées à chaque échantillon de test
    :param wl: longueurs d'ondes
    :param wlrdp: longueurs d'ondes passées par rdp
    :param savgol: booléen si on utilise le pré-traitement Savitzky Golay
    :param ref: si msc = True alors on utilise cette référence pour nos données de test
    :param rdp: booléen si on utilise le pré-traitement RDP
    :param justbymoda: booléen on sépare les classes par moda, ou par moda,feuille,das
    :param plot: booléen si on veut un affichage ou non
    :return: une nouvelle liste de reflectances, une nouvelle liste Y de classes (binaire)
    """
    X_test = X_test.astype('float32')
    if savgol is True:
        X_test = pre_traitement.lissageSavitzky(X_test, wl)
    if ref is not None:
        X_test = pre_traitement.compute_msc(X_test,wl, reference=ref, plot=plot)[0]
    if rdp is True:
        X_test = pre_traitement.computeRdpForTesting(wlrdp, X_test)
    X = getDatatoDataframe(X_test)
    if split:
        lb = preprocessing.MultiLabelBinarizer()
        if justbymoda is True:
            lb = preprocessing.LabelBinarizer()
        Y = lb.fit_transform(Y_test)
        #for i in range(0,Y.shape[0]):
            #print("Binarized labels test: "+str(Y[i])+" - "+str(Y_test[i]))
    else:
        Y=Y_test
    return X, Y

def giveClasseOfTest(Y_pred, Y_test, oneShotDictionary):
    """
    fonction qui permet de retourner la classes des données prédites.
    :param Y_pred: liste des classes prédites
    :return: liste des modas prédits
    """

#    print("giveClasse: "+str(oneShotDictionary))
    y_pred = []
    sample = 1
    for i in range(0,Y_pred.shape[0]):
        #print("Remapping: "+oneShotDictionary[str(i)])
        #print("Remapping: "+str(Y_pred[i]))
        #idx = (np.abs(Y_pred[i][-7:] - 1)).argmin()
        index = 0
        for j in range(0,Y_pred[i].shape[0]):
            if abs(1-Y_pred[i][j]) < abs(1-Y_pred[i][index]):
                index = j
        #print(index)
        binarizedLabel = []
        for j in range(0,Y_pred[i].shape[0]):
            binarizedLabel.append(0)
        #print(str(binarizedLabel))
        binarizedLabel[index] = 1
        #print(str(binarizedLabel)+" -> "+str(binarizedLabel).replace(",",""))
        dict = {'sample':sample}

        #print(oneShotDictionary[str(binarizedLabel).replace(",","")])
        dict['moda'] = oneShotDictionary[str(binarizedLabel).replace(",","")]
        #if (idx==0):
         #   dict['moda'] = 'FULL'
          #  y_pred.append(dict)
        #if (idx == 1):
         #   dict['moda'] = 'N0'
          #  y_pred.append(dict)
        #if (idx == 2):
         #   dict['moda'] = 'S0'
        y_pred.append(dict)
        y_pred.append(Y_test[sample-1])
        sample+=1
    return y_pred

def predicte(clf, X_test, Y_test, oneShotDictionary):
    """
    fonction qui permet d'afficher le moda associé aux échantillons de test
    :param clf: classifieur (ici regression pls)
    :param X_test: liste des réflectances de tests à prédire
    """
    y_pred = clf.predict(X_test)
    print("Moda prédit pour chaque echantillon de Test")
    for i in giveClasseOfTest(y_pred, Y_test, oneShotDictionary):
        print(i)

def initialisation(datacsv, X_test,justbymoda, split, size_test):
    """
    fonction qui permet d'initialiser les données d'entrainement et de test
    :param datacsv: les données importées depuis le csv
    :param X_test: données de test si pas de split
    :param justbymoda: booleen si on veut prédire juste en fonction du moda
    :param split: booleen si on veut séparer les données pour tester la confiance du classifieur
    :param size_test: la taille du test
    :return: données d'entrainement et de test
    """
    X = pd.DataFrame(datacsv).values[1:,4:]
    #[1:4] represente la colonne 2,3,4
    y = pd.DataFrame(datacsv).values[1:, 1:4]
    if justbymoda is True:
        y = pd.DataFrame(datacsv).values[1:, 1]
    if split:
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=size_test, shuffle=True)
    else:
        X_train = X
        Y_train = y
        X_test = X_test
        Y_test = None
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    datacsv = fonctionnalite.readCSV(sys.argv[1])
    testcsv = fonctionnalite.readCSV('data/sample_test.csv')

    wl = np.array(datacsv[0][4:])
    #si on veut séparer le jeu de données en train et test, pour tester le classifieur
    split = True

    # 20% données de test si on veut split
    size_test = 0.10

    # Si split est False on prend un exemple de test :
    X_test = pd.DataFrame(testcsv).values[1:]
    # si on veut faire un pré-traitement
    savgol = False
    msc = False
    rdp= False
    # si on veut classer juste en fonction des moda (mettre True), mettre False si on veut classer par moda,das,feuille
    justbymoda = True


    X_train, X_test, Y_train, Y_test = initialisation(datacsv,X_test=X_test,justbymoda=justbymoda,split=split, size_test=size_test)
    oneShotDictionary, X_train , Y_train, ref, wlrdp = prepareTraining(X_train,Y_train,wl=wl,savgol=savgol, msc=msc, rdp=rdp, justbymoda=justbymoda, plot=False)


    print(str(X_train.shape))

    counter = 0
    dataCounter = 0
    for wls in wlrdp:
        counter = counter + 1
    for d in datacsv:
        dataCounter = dataCounter + 1
    print(counter)
    print(dataCounter)

    labels = []
    for i in range(1,dataCounter):
        labels.append(datacsv[i][1])

    X_test, Y_test_encoded = prepareTesting(X_test, Y_test,wl, split=split, wlrdp=wlrdp, ref=ref, savgol=savgol, rdp=rdp, justbymoda=justbymoda, plot=False)

    clf = prediction(X_train, Y_train, X_test, Y_test_encoded)

    #décommenter si on veut afficher les moda prédit de l'échantillon de test
    predicte(clf, X_test, Y_test, oneShotDictionary)
