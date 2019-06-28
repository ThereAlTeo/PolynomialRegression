import csv
import pandas as pd
import os.path as fileSystem

FILEPATH = "OnlineNewsPopularity.csv"

#Zona del programma in cui vengono collocate le funzioni.
#Esse verranno chiamate all'occorrenza all'interno del programma


def loadCSVFile(path):
    if fileSystem.exists(path):
        return pd.read_csv(path, sep=",")
    else:
        print("File non trovato")

#La funzione visualizza il dtype di ogni ottributo del dataFrame passatogli.
#Aggiunge infine anche l'occupazione in memoria.
def generalDataFrameInfo(dataFrame):
    dataFrame.info(memory_usage="deep")

def getRelativePath():
    return fileSystem.dirname(fileSystem.dirname(__file__)) + "\\res\\datasheet\\"

#INIZIO DEL PROGRAMMA

datasheet = loadCSVFile(str(getRelativePath()) + str(FILEPATH))

#print(datasheet)
generalDataFrameInfo(datasheet)
