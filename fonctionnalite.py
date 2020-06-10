#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd
#import plotly.tools as tls
import chart_studio.tools as tls
#import plotly.plotly as py
import chart_studio.plotly as py
import plotly.graph_objs as go


tls.set_credentials_file(username='????', api_key='?????')

listeDef = ['FULL', 'N0', 'S0']
listeFeuille = ['F3','F5']
minDas = 23

def organizeData(mat):
    """
    fonction qui transforme les données en dictionnaire
    :param mat: les données importées du csv
    :return: les données sous forme dictionnaire
    """
    print("\\\\\\\\\\\  Transform data to dict  ////////////")
    data = []
    for line in mat:
        dico = {'Sample': line[0], 'Moda': line[1], 'Feuille': line[2], 'DaS': line[3],
                'Reflectance': np.array(line[4:]).astype(np.float64)}
        data.append(dico)
    return data

def readCSV(filename):
    """
    fonction qui permet de lire un csv et ajoute les données dans une liste
    :param filename: le nom du csv à lire
    :return: liste des données
    """
    print("\\\\\\\\\\\  Lecture CSV  ////////////")
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            data += [[x for x in line]]
    return data

def readMatrix(data):
    """
    fonction qui permet de transformer les données du dictionnaire en matrice
    :param data: données dictionnaires
    :return: matrice des données sous format np
    """
    print("\\\\\\\\\\\  Lecture matrice  ////////////")
    matrix = []
    for sample in data:
        matrix.append(sample['Reflectance'])
    return np.stack(matrix)

def buildCSVtest(alldata,wl):
    """
    fonction qui construit un échantillon de test (imaginaire), en prenant la moyenne de tous les samples
    :param alldata: les données d'origine
    :param wl: les longueurs d'ondes
    """
    csv = open('data/sample_test.csv', 'w')
    features = []
    X = pd.DataFrame(alldata).values[1:, 4:]
    X = X.astype('float32')
    for i in wl:
        features.append(str(i))
    csv.write(';'.join(features) + '\n')
    csv.write(';'.join(np.mean(X, axis=0).astype('str')) + '\n')
    csv.close()

def reduceDate(data):
    """
    fonction qui réduit les données du tout début avec les bonnes données à analyser dans un csv
    :param data:  données instanciers
    """
    csvData = []
    csv = open('reduced_cleaned_spectra.csv', 'w')
    csv.write(';'.join(data[0]) + '\n')
    print(data[0])
    del data[0]
    for line in data:
        if(int(line[0])<=726 and str(line[2])!="F1" and int(line[3])>=23 and str(line[1]) in listeDef):
            csv.write(';'.join(line) + '\n')
    csv.close()



def getNewData(y):
    """
    fonction qui permet de récupérer les nouvelles données das>=23 et feuille F3 F5
    :param y: liste des différents échantillons
    :return: liste des variances
    """
    listeReflec = []
    for sample in y:
        if (int(sample['DaS'])>=minDas and sample['Feuille'] in listeFeuille):
            listeReflec.append(sample)
    return listeReflec

def plotModaByDasAndFeuille(y):
    """
    fonction qui permet d'afficher graphiquement les nouveaux échantillons par rapport aux données globales (listeDef, minDas..)
    :param y: liste des différents échantillons
    """
    print("\\\\\\\\\\\  PLOTTING  ////////////")
    fig = tls.make_subplots(rows=1, cols=1,
                            shared_xaxes=True, shared_yaxes=True,
                            )
    listemoda = getNewData(y)

    for moda in listeDef:
        xaxis = []
        yaxis = []
        for sample in listemoda:
            if(sample['Moda']==moda):
                for i in range(0,1550):
                    xaxis.append(i+450)
                    yaxis.append((sample['Reflectance'][i]))
        fig.append_trace(
                go.Scatter(x=xaxis, y=yaxis, mode="markers", name=moda), 1, 1)
    fig.layout.update({'title': 'Full,N0,S0, Das>=27 et F3-F5 '}, height=800, width=1000)
    fig.layout.yaxis.update(title='Reflectance')
    fig.layout.xaxis.update(title='Longueur d\'onde', range=[450, 2000], tick0=450, dtick=150)
    plot_url = py.plot(fig, filename='Full,N0,S0, Das>=27 et F3-F5 ')

def dataToCssv(data,wl,features,savgol=False, msc=False, rdp= False):
    """
    fonction qui permet de crer un csv des données pré traitées
    :param data: les données pre-traitées
    :param wl: longueur d'ondes
    :param features: caractéristiques
    :param savgol: booleen, si on a appliqué le filtre Savitzky Golay
    :param msc: booleen, si on a appliqué MSC
    :param rdp: booleen, si on a appliqué RDP
    """
    str_rdp = ""
    str_savgol = ""
    str_msc = ""
    if rdp:
        str_rdp = "_rdp"
    if savgol:
        str_savgol = "_savgol"
    if msc:
        str_msc = "_msc"

    csv = open('data/reduced_cleaned_spectra'+str_savgol+''+str_msc+''+str_rdp+'.csv', 'w')
    for i in wl:
        features.append(str(i))
    csv.write(';'.join(features) + '\n')
    del data[0]
    for line in data:
        csv.write(';'.join(line) + '\n')
    csv.close()

# Décommenter la partie d'en dessous si on veut générer un pré traitement sur les données d'origines

# file = 'data/reduced_cleaned_spectra.csv'
#
# alldata = readCSV(file)
# savgol = True
# msc = True
# rdp = True
# features = alldata[0][:4]
# wl = np.array(alldata[0][4:])
# alldata, wl = pre_traitement.preProcessing(wl,alldata,savgol=savgol, msc=msc, rdp=rdp)
# dataToCssv(alldata,wl,features,savgol=savgol,msc=msc,rdp=rdp)
