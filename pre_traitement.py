#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp
#import plotly.tools as tls
import chart_studio.tools as tls
#import plotly.plotly as py
import chart_studio.plotly as py
import plotly.graph_objs as go

listeDef = ['FULL', 'N0', 'S0', 'Nint', 'Sint', 'P0', 'Pint', 'FULLP']
listeFeuille = ['F1', 'F3','F5']
minDas = 13



def getReflectanceByDeficiency(data, moda, Y, justbymoda):
    """
    fonction qui permet de récupérer les reflectances par moda
    :param data: données instanciers
    :param moda: moda à calculer
    :param  Y: classe des données instanciers
    :param justbymoda: booleen associé a la regression pls
    :return:
    """
    matrice = []
    for i in range(0,len(Y)):
        if justbymoda is True:
            if (Y[i] == moda):
                matrice.append(data[i])
        else:
            if (Y[i][0] == moda):
                matrice.append(data[i])
    return np.stack(matrice, axis=1)


def computeMeanByModa(data,moda, Y, justbymoda):
    """
    fonction qui calcule la moyenne par moda
    :param data: données instanciers
    :param moda: moda à calculer
    :param  Y: classe des données instanciers
    :param justbymoda: booleen associé a la regression pls
    :return:
    """
    matrice = getReflectanceByDeficiency(data, moda, Y, justbymoda)
    meanModa = []
    for onde in matrice:
        mean = 0
        for reflex in onde:
            mean += reflex
        meanModa.append(mean / len(onde))
    return np.array(meanModa)

def computeRdpForMean(wl, data, Y, justbymoda):
    """
    fonction qui permet de calculer le rdp associé à la moyenne des reflectances par modas, on peut baisser ou augmenter
    le seuil pour augmenter ou baisser respectivement le nombre de longueur d'onde
    :param wl: longueurs d'onde
    :param data: données instanciers
    :param Y: classe des données instanciers
    :param justbymoda: booleen associé a la regression pls
    :return: liste des données rdp
    """
    meandef = []
    for deficiency in listeDef:
        meandef.append({'Moda':deficiency, 'Reflectance':computeMeanByModa(data,deficiency,Y, justbymoda)})
    for meansample in meandef:
        i = 0
        listeCouple = []
        while i < len(wl):
            listeCouple.append([float(wl[i]), meansample['Reflectance'][i]])
            i += 1
        meansample['Reflectance']=rdp(listeCouple, epsilon=0.001)
    return meandef

def computeRdpForTesting(listeWl, data):
    """
    fonction qui permet d'appliquer rdp aux données de test
    :param listeWl: la liste des longueurs d'onde créée par les données d'entrainement
    :param data: liste des donnée de test
    :return: la nouvelle liste des données de test avec rdp appliqué
    """
    new_data = []
    for sample in data:
        tmp = []
        for datax in listeWl:
            tmp.append(sample[datax-450])
        new_data.append(tmp)
    return new_data




def lissageSavitzky(data, wl, plot=False):
    """
    fonction qui permet d'appliquer le filtre de Savitzky-Golay
    :param data: données instanicers
    :param wl: longueurs d'onde
    :param plot: booleen si on veut afficher la différence entre les données instanicers et celle de rdp
    :return: les données filtré par Savitzky-Golay
    """
    test = data[0].copy()

    #print("lissageSavitzky: "+str(len(data))+" "+str(len(data[0])))

    test2 = savgol_filter(data[0],11,3)

    if plot is True:
        print("\\\\\\\\\\\  PLOTTING  ////////////")
        fig = tls.make_subplots(rows=1, cols=1,
                                shared_xaxes=True, shared_yaxes=True,
                                )
        fig.append_trace(
            go.Scatter(x=wl, y=test, name='Donnée'), 1, 1)
        fig.append_trace(
            go.Scatter(x=wl, y=test2, name='Donnée filtrée'), 1, 1)
        fig.layout.update({'title': 'Filtre de Savitzky'}, height=800, width=1000)
        fig.layout.yaxis.update(title='Reflectance')
        fig.layout.xaxis.update(title="Longueur d'onde")
        plot_url = py.plot(fig, filename='Filtre de Savitzky')
    for sample in data:
        sample = savgol_filter(sample, 11, 3)
    return data




def compute_msc(data,wl, reference=None, plot=False):
    """
    fonction qui permet de calculer la simplification Multiplicative scatter correction sur les données
    :param data: donnée instancier
    :param wl: longueurs d'onde
    :param reference: la moyenne des données d'origines pour simplifier les données de test
    :param plot: booleen si on veut afficher la différence entre les données instanicers et celle de msc
    :return: datamsc : les données transformées par msc, ref : la moyenne de reflectance en reférence des données
    d'origines
    """
    if reference is None:
        ref = np.mean(data, axis=0)
    else:
        ref = reference
    data_msc = np.zeros_like(data)

    for i in range(data.shape[0]):
        fit = np.polyfit(ref, data[i, :], 1, full=True)
        data_msc[i, :] = (data[i, :] - fit[0][1]) / fit[0][0]+0.1


    if plot:
        print("ploting MSC")
        plt.figure(figsize=(8, 9))
        ax1 = plt.subplot(211)
        plt.plot(wl.astype('int'), data.T)
        plt.title('Original data')
        plt.ylabel('Reflectance')
        plt.xlim(450,2000)
        ax2 = plt.subplot(212)
        plt.plot(wl.astype('int'), data_msc.T)
        plt.ylabel('Reflectance')
        plt.title('MSC')
        plt.xlabel("Longueur d'onde")
        plt.xlim(450,2000)
        plt.show()
    return (data_msc, ref)

def computeRdpForData(wl, data, Y, justbymoda, plot=False):
    """
    fonction qui permet de simplier les données instanciers via l'algorithme Ramer-Douglas-Peucker
    :param wl: longueurs d'onde
    :param data: donnée d'origine
    :param Y: classe des données instanciers
    :param justbymoda: booleen associé a la regression pls
    :param plot: si on veut afficher la différence entre un sample et un sample somplifié par rdp
    :return: listeWl: nouvelle liste restante de longueurs d'ondes après RDP, new_data : nouvelle liste de données RDP
    """
    meandef = computeRdpForMean(wl, data, Y, justbymoda)
    listeWl = []
    for meansample in meandef:
        for couple in meansample['Reflectance']:
            if(couple[0] not in listeWl):
                listeWl.append(int(couple[0]))
    listeWl.sort()
    test = data[0]
    testrdp = []
    for datax in listeWl:
        testrdp.append(data[0][datax-450])

    rdp = np.array(testrdp).astype(np.float32)
    new_data = []
    for sample in data:
        tmp = []
        for datax in listeWl:
            tmp.append(sample[datax-450])
        new_data.append(tmp)

    if plot:
        print("ploting RDP")
        plt.figure(figsize=(8, 9))
        ax1 = plt.subplot(211)
        plt.plot(wl.astype('int'), test,'.')
        plt.title('Original data')
        plt.ylabel('Reflectance')
        plt.xlim(450,2000)
        ax2 = plt.subplot(212)
        plt.plot(listeWl, rdp,'.')
        plt.ylabel('Reflectance')
        plt.title('RDP')
        plt.xlabel("Longueur d'onde")
        plt.xlim(450,2000)

        plt.show()
    return listeWl, new_data

def preProcessing(wl,data, savgol=False, msc=False, rdp = False):
    """
    fonction qui permet de faire le pre-traitement des données via des options
    :param wl: longueurs d'onde
    :param data: donnée d'origine
    :param savgol: booleen si on veut appliquer le filtre Savitzky Golay
    :param msc: booleen si on veut appliquer MSC
    :param rdp: booleen si on veut appliquer RDP
    :return: data_process : nouvelle donnée pre-traité, wl : longueurs d'ondes restantes
    """
    sample = pd.DataFrame(data).values[1:,0]
    features = pd.DataFrame(data).values[0,:4]
    X = pd.DataFrame(data).values[1:, 4:]
    y = pd.DataFrame(data).values[1:, 1:4]
    X = X.astype('float32')
    if savgol is True:
        X = lissageSavitzky(X,wl)
    if msc is True:
        X = compute_msc(X,wl)[0]
    if rdp is True:
        wl, X = computeRdpForData(wl, X, y, False)
    data_pross = []
    list_features = features.tolist()
    for i in wl:
        list_features.append(str(i))
    data_pross.append(list_features)
    for i in range(0,len(X)):
        tmp = []
        tmp.append(sample[i])
        for j in y[i]:
            tmp.append(j)
        for j in X[i]:
            tmp.append(str(j))

        data_pross.append(tmp)
    return data_pross, wl





