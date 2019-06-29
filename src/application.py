import csv
import pandas as pd
import os.path as fileSystem
import matplotlib.pyplot as plot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge

FILEPATH = "OnlineNewsPopularity.csv"

'''Zona del programma in cui vengono collocate le funzioni.
Esse verranno chiamate all'occorrenza all'interno del programma'''
def loadCSVFile(path):
    if fileSystem.exists(path):
        return pd.read_csv(path, sep=",")#, dtype={x: "bool" for x in range(30, 39)})
    else:
        print("File non trovato")

'''
Restituisce l'indice di correlazione tra due feature.
param:
feature1, feature2: nparray or series
returns:
indice fi correlazione
'''
def getCorrelation(feature1, feature2):
        return np.mean((feature1-feature1.mean()) * (feature2-feature2.mean())) / (feature1.std() * feature2.std())

'''La funzione visualizza il dtype di ogni ottributo del dataFrame passatogli.
Aggiunge infine anche l'occupazione in memoria.'''
def generalDataFrameInfo(dataFrame):
    dataFrame.info(memory_usage="deep")

'''La funzione crea in grafico a dispersione'''
def showDispersionGraph(feature1, feature2):
    import matplotlib.pyplot as plt
    plt.scatter(feature1, feature2)
    plt.show()
    return 0

def getRelativePath():
    return fileSystem.dirname(fileSystem.dirname(__file__)) + "\\res\\datasheet\\"

'''La funzione describe tende ad escludere il primo attributo.
Probabilmente perchè di tipo object e quindi non ha competenze per il calcolo dei valori.'''
def exploratoryAnalysis(dataFrame):
    generalDataFrameInfo(dataFrame)
    print(dataFrame.describe())

    pd.cut(dataFrame[" rate_positive_words"], 4).value_counts().plot.pie()
    plot.show()

def elaborationWithoutLasso(XTrain, YTrain):
    prm = Pipeline([("poly",   PolynomialFeatures(degree=2, include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", LinearRegression(normalize=True))])
    prm.fit(XTrain, YTrain)
    return prm

def elaborationWithLasso(XTrain, YTrain):
    model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", Lasso(alpha=2, tol=0.001 ,max_iter = 2000))])

    model.fit(XTrain, YTrain)

    return model

def dataElaboration(dataFrame):
    Y = dataFrame[" shares"].values
    X = dataFrame.drop([" shares", "url", " timedelta" ," n_tokens_title", " n_tokens_content", " n_unique_tokens"," n_non_stop_words"," n_non_stop_unique_tokens"," num_hrefs"," num_self_hrefs"," num_imgs"," num_videos", " average_token_length", " num_keywords", " data_channel_is_lifestyle", " data_channel_is_entertainment", " data_channel_is_bus", " data_channel_is_socmed", " data_channel_is_tech", " data_channel_is_world" , " kw_min_min", " kw_max_min", " kw_avg_min", " kw_min_max" ," kw_max_max", " kw_avg_max" , " kw_min_avg", " kw_max_avg" , " kw_avg_avg" , " self_reference_min_shares"], axis=1)
    XTrain, XVal, YTrain, YVal = slipDataset(X, Y)
    p = elaborationWithoutLasso(XTrain, YTrain)
    print(p)
    print(pd.Series(p.named_steps["linreg"].coef_, XTrain.columns))

def slipDataset(X, Y):
    return train_test_split(X, Y, test_size=1/10, random_state=73)

#INIZIO DEL PROGRAMMA

#Da verificare il corretto utilizzo di datasheet.
#Esso può essere richiamato e utilizzato dalla funzione, senza l'obbligo di essere passato alle funzioni stesse come argomento.
#Può essere considerato con scope globale all'interno del progetto.
dataset = loadCSVFile(str(getRelativePath()) + str(FILEPATH))

'''ANALISI ESPLORATIVA'''
exploratoryAnalysis(dataset)
dataElaboration(dataset)
