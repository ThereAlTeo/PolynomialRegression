import csv
import pandas as pd
import os.path as fileSystem
import matplotlib.pyplot as plot

FILEPATH = "OnlineNewsPopularity.csv"

'''Zona del programma in cui vengono collocate le funzioni.
Esse verranno chiamate all'occorrenza all'interno del programma'''

def loadCSVFile(path):
    if fileSystem.exists(path):
        return pd.read_csv(path, sep=",")#, dtype={x: "bool" for x in range(30, 39)})
    else:
        print("File non trovato")

'''La funzione visualizza il dtype di ogni ottributo del dataFrame passatogli.
Aggiunge infine anche l'occupazione in memoria.'''
def generalDataFrameInfo(dataFrame):
    dataFrame.info(memory_usage="deep")

def getRelativePath():
    return fileSystem.dirname(fileSystem.dirname(__file__)) + "\\res\\datasheet\\"

'''La funzione describe tende ad escludere il primo attributo.
Probabilmente perchè di tipo object e quindi non ha competenze per il calcolo dei valori.'''
def exploratoryAnalysis(dataFrame):
    generalDataFrameInfo(dataFrame)
    print(dataFrame.describe())

    pd.cut(dataFrame[" rate_positive_words"], 4).value_counts().plot.pie()
    plot.show()

#INIZIO DEL PROGRAMMA

#Da verificare il corretto utilizzo di datasheet.
#Esso può essere richiamato e utilizzato dalla funzione, senza l'obbligo di essere passato alle funzioni stesse come argomento.
#Può essere considerato con scope globale all'interno del progetto.
datasheet = loadCSVFile(str(getRelativePath()) + str(FILEPATH))

'''ANALISI ESPLORATIVA'''
exploratoryAnalysis(datasheet)
