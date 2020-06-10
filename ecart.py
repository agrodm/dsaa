#!/usr/bin/env python
# -*- coding: utf-8 -*-
import plotly.tools as tls
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import fonctionnalite

tls.set_credentials_file(username='?????', api_key='??????')
listeDef = fonctionnalite.listeDef
listeFeuille = fonctionnalite.listeFeuille
minDas = fonctionnalite.minDas



def computeMeanByModa(y, moda):
    """
    fonction qui permet de calculer la moyenne de reflectance par moda
    :param y: liste des différents échantillons
    :param moda: string : moda
    :return: liste de la moyenne de réflectance
    """
    matrice = getReflectanceByDeficiency(y, moda)
    meanModa = []
    for onde in matrice:
        mean = 0
        for reflex in onde:
            mean += reflex
        meanModa.append(mean / len(onde))
    return np.array(meanModa)

def getListCombinaison(seq, k):
    """
    fonction qui permet de récupérer l'ensemble des parties d'un ensemble
    :param seq: liste d'objet
    :param k: longueur de l'ensemble
    :return: liste d'ensemble
    """
    p = []
    i, imax = 0, 2 ** len(seq) - 1
    while i <= imax:
        s = []
        j, jmax = 0, len(seq) - 1
        while j <= jmax:
            if (i >> j) & 1 == 1:
                s.append(seq[j])
            j += 1
        if len(s) == k:
            p.append(s)
        i += 1
    return p

def searchDeficiency(y):
    """
    fonction qui permet de récupérer la liste des modas
    :param y: liste des différents échantillons
    :return: liste des modas
    """
    listeDeficiency = []
    for sample in y:
        if sample['Moda'] not in listeDeficiency:
            listeDeficiency.append(sample['Moda'])
    return listeDeficiency

def searchDas(y):
    """
    fonction qui permet de récupérer la liste des das
    :param y: liste des différents échantillons
    :return: liste des das
    """
    listeDeficiency = []
    for sample in y:
        if sample['DaS'] not in listeDeficiency:
            listeDeficiency.append(sample['DaS'])
    return listeDeficiency

def searchFeuille(y):
    """
    fonction qui permet de récupérer la liste des feuilles
    :param y: liste des différents échantillons
    :return: liste des feuilles
    """
    listeDeficiency = []
    for sample in y:
        if sample['Feuille'] not in listeDeficiency:
            listeDeficiency.append(sample['Feuille'])
    return listeDeficiency

def computeMeanByCriteria(y, moda, feuille, das):
    """
    fonction qui permet de calculer la moyenne par critère
    :param y: liste des différents échantillons
    :param moda: string
    :param feuille: string
    :param das: string
    :return: liste des moyennes de réflectances
    """
    print("\\\\\\\\\\\  Compute mean of " + feuille + ", moda=" + moda + " and DaS=" + das + "////////////")
    i = 0
    matrice = []
    for sample in y:
        if (sample['Feuille'] == feuille and sample['DaS'] == das and sample['Moda'] == moda):
            matrice.append(sample['Reflectance'])
    matrice = np.stack(matrice, axis=1)
    meanModa = []
    for onde in matrice:
        mean = 0
        for reflex in onde:
            mean += reflex
        meanModa.append(mean / len(onde))
    return np.array(meanModa)

def plotVarianceComparaisons(x, y):
    '''
    fonction qui permet d'afficher graphiquement la variance entre les différents modas
    :param x: longueur d'ondes
    :param y: liste des différents échantillons
    '''
    print("\\\\\\\\\\\  PLOTTING  ////////////")
    listeDeficiency = searchDeficiency(y)
    listeDeficiencyCombin = getListCombinaison(listeDeficiency, 2)
    fig = tls.make_subplots(rows=7, cols=4, subplot_titles=([x[0] + "-" + x[1] for x in listeDeficiencyCombin]))
    row = 1
    col = 0
    deficiencyColor = {"FULL": 'green', "FULLP": 'blue', "N0": 'red', "Nint": 'yellow', "S0": 'pink', "Sint":'purple', "P0":'brown',"Pint":'orange'}
    deficiencyLegended = []
    for couple in listeDeficiencyCombin:
        col += 1
        findOndesSignificatives(y,couple[0],couple[1])
        for deficiency in couple:
            if deficiency not in deficiencyLegended:
                deficiencyLegended+=[deficiency]
                fig.append_trace(
                    go.Scatter(x=x, y=computeVarianceByModa(y, deficiency), legendgroup=deficiency,
                               name=deficiency, mode='markers', marker={'color': deficiencyColor[deficiency]}), row,
                    col)
            else:
                fig.append_trace(
                    go.Scatter(x=x, y=computeVarianceByModa(y, deficiency), legendgroup=deficiency,showlegend=False,
                               name=deficiency, mode='markers', marker={'color': deficiencyColor[deficiency]}), row, col)
        if col == 4:
            col = 0
            row += 1
    fig.layout.update({'title': '28 Comparaisons des variances'}, height=800, width=1400)
    plot_url = py.plot(fig, filename='28 comparaisons variances')


def getReflectanceByDeficiency(y, moda):
    """
    fonction qui permet de récupérer les données de reflectances par rapport à un moda
    :param y: liste des différents échantillons
    :param moda: string
    :return: liste des reflectances
    """
    matrice = []
    for sample in y:
        if (sample['Moda'] == moda):
            matrice.append(sample['Reflectance'])
    return np.stack(matrice, axis=1)





def computeVarianceByModa(y, moda):
    """
    fonction qui permet de calculer la variance d'un moda
    :param y: liste des différents échantillons
    :param moda: string
    :return: liste des variances
    """
    matrice = getReflectanceByDeficiency(y, moda)
    meanModa = []
    for onde in matrice:
        meanModa.append(np.std(onde)**2)
    return np.array(meanModa)




def computeVarianceByModaCriteria(y, moda,feuille , das):
    """
    fonction qui permet de calculer la variance par critère sur les données
    :param y: liste des différents échantillon
    :param moda: string  : moda à chercher
    :param feuille: string : feuille à chercher
    :param das: string : das à chercher
    :return: liste des variances
    """
    matrice = getReflectanceByDefiencyCriteria(y,moda,feuille,das)
    meanModa = []
    for onde in matrice:
        meanModa.append(np.std(onde)**2)
    return np.array(meanModa)

def computeEcartTypeByModaCriteria(y, moda,feuille , das):
    """
    fonction qui permet de calculer l'écart type par critère sur les données
    :param y: liste des différents échantillons
    :param moda: string  : moda à chercher
    :param feuille: string : feuille à chercher
    :param das: string : das à chercher
    :return: liste des écarts types
    """
    matrice =  getReflectanceByDefiencyCriteria(y,moda,feuille,das)
    meanModa = []
    for onde in matrice:
        meanModa.append(np.std(onde))
    return np.array(meanModa)

def findOndesSignificatives(data, moda1, moda2):
    """
    fonction qui permet de récupérer les ondes significatives entre 2 modas
    :param data: liste des différents échantillons
    :param moda1: string : premier moda
    :param moda2: string : deuxième moda
    :return: liste des ondes
    """
    ondes=[]
    ecartMoyenModa1Moda2=ecartMoyen(data,moda1,moda2)
    reflectancesModa1 = computeMeanByModa(data, moda1)
    reflectancesModa2 = computeMeanByModa(data, moda2)
    for i in range (0,len(reflectancesModa1)):
        if ecartMoyenModa1Moda2*1.5 < abs(reflectancesModa1[i]- reflectancesModa2[i]):
            ondes+=[i+450]
    print("Ondes significatives entre "+ moda1 + " et " + moda2 + " = " + str(ondes))
    return ondes



def gapComparaisons(y):
    """
    fonction qui permet d'afficher la comparaison des ondes significatives par rapport au moda FULL
    :param y: liste de différents échantillons
    """
    listeDeficiency = searchDeficiency(y)
    listeDeficiencyCombin = getListCombinaison(listeDeficiency, 2)
    newOndes=[]
    for couple in listeDeficiencyCombin:
        if couple[0] == "FULL" or couple[1] == "FULL":
            ondesToKeep=findOndesSignificatives(y,couple[0],couple[1])
            for onde in ondesToKeep:
                if onde not in newOndes:
                    newOndes.append(onde)
    newOndes.sort()
    print("Il y a " + str(len(newOndes)) + " ondes significatives en tout: ")
    print(newOndes)

def VarianceEcartTyeComparaisons(y):
    """
    fonction qui permet d'afficher la variance et écart type de chaque moda
    :param y: liste des différents échantillons
    """
    listeDeficiency = searchDeficiency(y)
    for deficiency in listeDeficiency:
        couples = []
        for sample in y:
            if sample["Moda"] == deficiency and sample["Feuille"]+sample["DaS"] not in couples:
                print(deficiency + " "+ sample["Feuille"]+ " "+ sample["DaS"] + ", ecart type = "+ str(computeEcartTypeByModaCriteria(y,sample["Moda"],sample["Feuille"],sample["DaS"]))
                      +"\n variance = "+str(computeVarianceByModaCriteria(y,sample["Moda"],sample["Feuille"],sample["DaS"])))
                couples.append(sample["Feuille"]+sample["DaS"])


def plotVarianceComparaisonsByCriteria(x, y):
    """
    fonction qui permet d'afficher graphiquement les variances entre chaque moda
    :param x: longueurs d'ondes
    :param y: liste des différents échantillons
    """
    print("\\\\\\\\\\\  PLOTTING  ////////////")
    listeDeficiency = searchDeficiency(y)
    fig = tls.make_subplots(rows=8, cols=1, subplot_titles=listeDeficiency)
    couples = []
    row = 1
    col = 1
    for deficiency in listeDeficiency :
        for sample in y :
            if sample["Moda"] == deficiency and sample["Moda"]+ sample["Feuille"] + sample["DaS"] not in couples:
                fig.append_trace(
                go.Scatter(x=x, y=computeVarianceByModaCriteria(y,sample["Moda"],sample["Feuille"],sample["DaS"]), showlegend=False, legendgroup=str(sample["Feuille"] + sample["DaS"]),
                           name=str(sample["Moda"]+"-"+ sample["Feuille"] +"-D"+sample["DaS"]), mode='markers'), row, col)
                couples.append(sample["Moda"]+ sample["Feuille"] + sample["DaS"])
        row+=1
    fig.layout.update({'title': 'Comparaisons des variances'}, height=2000, width=1400)
    plot_url = py.plot(fig, filename='comparaisons variances')

def groupByDas(y, das):
    """
    fonction qui permet de grouper les reflectances par das
    :param y: liste des différents échantillons
    :param das: string : das à chercher
    :return: liste des reflectances
    """
    listeReflec = []
    for sample in y:
        if (sample['Das'] == das):
            listeReflec.append(sample)
    return listeReflec

def groupByFeuille(y,feuille):
    """
    fonction qui permet de grouper les reflectances par feuille
    :param y: liste des différents échantillons
    :param feuille: string : feuille à chercher
    :return: liste des reflectances
    """
    listeReflec = []
    for sample in y:
        if (sample['Feuille'] == feuille):
            listeReflec.append(sample)
    return listeReflec



def ecartMoyen(data,moda1,moda2):
    """
    fonction qui permet de récupérer et afficher l'écart moyen entre 2 modas
    :param data: liste des différents échantillons
    :param moda1: string : premier moda
    :param moda2: string : deuxième moda
    :return: ecart moyen
    """
    ecart=0
    reflectancesModa1 = computeMeanByModa(data, moda1)
    reflectancesModa2 = computeMeanByModa(data, moda2)
    for i in range (0,len(reflectancesModa1)):
        ecart+=abs(reflectancesModa1[i]- reflectancesModa2[i])
    print("Moyenne de l'ecart entre "+ moda1 + "et" + moda2 + " = " +str(ecart/len(reflectancesModa1)))
    return ecart/len(reflectancesModa1)

def getReflectanceByDefiencyCriteria(y, moda,feuille , das):
    """
    fonction qui permet de récupérer la reflectance en fonction des critères qu'on lui donne
    :param y: liste des différents échantillons
    :param moda: string : moda à chercher
    :param feuille: string : feuille à chercher
    :param das: string : das à chercher
    :return: liste des reflectances
    """
    matrice = []
    for sample in y:
        if (sample['Feuille'] == feuille and sample['DaS'] == das and sample['Moda'] == moda):
            matrice.append(sample['Reflectance'])
    return np.stack(matrice, axis=1)

def computeEcartTypeByModa(y, moda):
    """
    fonction qui permet de calculer l'écart type par moda
    :param y: liste des différents échantillons
    :param moda:  string: moda
    :return: liste des écarts types
    """
    matrice = getReflectanceByDeficiency(y, moda)
    meanModa = []
    for onde in matrice:
        meanModa.append(np.std(onde))
    return np.array(meanModa)

def plotMeans(x, y):
    """
    fonction qui permet d'afficher graphiquement la moyenne pour chacun des modas instanciers dans listeDef
    :param x: longueurs d'onde
    :param y: liste des différents échantillons
    """
    print("\\\\\\\\\\\  PLOTTING  ////////////")
    fig = tls.make_subplots(rows=1, cols=1,
                            shared_xaxes=True, shared_yaxes=True,
                            )

    for deficiciency in listeDef:
        fig.append_trace(
            go.Scatter(x=x, y=computeMeanByModa(y, deficiciency), name=deficiciency), 1, 1)

    fig.layout.update({'title': 'Moyenne pour FULL,N0,S0'}, height=800, width=1000)
    fig.layout.yaxis.update(title='Reflectance')
    fig.layout.xaxis.update(title='Longueur d\'onde', range=[450, 2000], tick0=450, dtick=150)

    plot_url = py.plot(fig, filename='Moyenne pour FULL,N0,S0')


# décommenter pour tester

# file = 'data/cleaned_spectra.csv'
# alldata = fonctionnalite.readCSV(file)
#
# wl = np.array(alldata[0][4:])
# data = fonctionnalite.organizeData(alldata)
#
#
# #exemple de fonctions
# gapComparaisons(data)
# VarianceEcartTyeComparaisons(data)



