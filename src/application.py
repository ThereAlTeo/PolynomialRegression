import csv
import pandas as pd
import numpy as np
import os.path as fileSystem
import matplotlib.pyplot as plot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


FILEPATH = "DataSheet.csv"


'''Zona del programma in cui vengono collocate le funzioni.
Esse verranno chiamate all'occorrenza all'interno del programma'''
def loadCSVFile(path):
    if fileSystem.exists(path):
        return pd.read_csv(path, sep=",", dtype={x: "category" for x in range(2, 10)})
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


'''binarizza le feature categoriche'''
def binarizza(dataset):
    dataset["Age0-17"] = np.where(dataset["Age"] == "0-17", 1, 0)
    dataset["Age18-25"] = np.where(dataset["Age"] == "18-25", 1, 0)
    dataset["Age26-35"] = np.where(dataset["Age"] == "26-35", 1, 0)
    dataset["Age36-45"] = np.where(dataset["Age"] == "36-45", 1, 0)
    dataset["Age46-50"] = np.where(dataset["Age"] == "46-50", 1, 0)
    dataset["Age51-55"] = np.where(dataset["Age"] == "51-55", 1, 0)
    dataset["Age55+"] = np.where(dataset["Age"] == "55+", 1, 0)
    dataset["GenderF"] = np.where(dataset["Gender"] == "F", 1, 0)
    dataset["GenderM"] = np.where(dataset["Gender"] == "M", 1, 0)
    dataset["CityA"] = np.where(dataset["CityCategory"] == "A", 1, 0)
    dataset["CityB"] = np.where(dataset["CityCategory"] == "B", 1, 0)
    dataset["CityC"] = np.where(dataset["CityCategory"] == "C", 1, 0)
    dataset["StayInCurrentCityYears"] = np.where(dataset["StayInCurrentCityYears"] == "4+", 4, dataset["StayInCurrentCityYears"])
    dataset["StayInCurrentCityYears"] = np.where(dataset["StayInCurrentCityYears"] == "3", 3, dataset["StayInCurrentCityYears"])
    dataset["StayInCurrentCityYears"] = np.where(dataset["StayInCurrentCityYears"] == "2", 2, dataset["StayInCurrentCityYears"])
    dataset["StayInCurrentCityYears"] = np.where(dataset["StayInCurrentCityYears"] == "1", 1, dataset["StayInCurrentCityYears"])
    pd.to_numeric(dataset["StayInCurrentCityYears"])
    return dataset.drop(["CityCategory", "Gender","Age"], axis=1)



def getRelativePath():
    return fileSystem.dirname(fileSystem.dirname(__file__)) + "\\res\\datasheet\\"


'''La funzione describe tende ad escludere il primo attributo.
Probabilmente perchè di tipo object e quindi non ha competenze per il calcolo dei valori.'''
def exploratoryAnalysis(dataFrame):
    generalDataFrameInfo(dataFrame)
    print(dataFrame.describe())
    pd.cut(dataFrame["MaritalStatus"], 2).value_counts().plot.pie()
    plot.show()


def elaborationWithoutLasso(XTrain, YTrain):
    prm = Pipeline([("poly",   PolynomialFeatures(degree=2, include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", LinearRegression(normalize=True))])
    prm.fit(XTrain, YTrain)
    return prm


def relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def elaborationWithLasso(XTrain, YTrain):
    model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", Lasso(alpha=1))])
    model.fit(XTrain, YTrain)
    return model


def dataElaboration(dataFrame):
    Y = dataFrame["Purchase"].values
    X = dataFrame.drop(["ProductCategory3","ProductCategory2", "Purchase"], axis=1)
    XTrain, XVal, YTrain, YVal = slipDataset(X, Y)
    p = elaborationWithLasso(XTrain, YTrain)
    #print(p)
    print(p.named_steps["linreg"].coef_)
    print(XTrain.columns)
    printEvalutation(XVal, YVal, p)
    #print(pd.Series(p.named_steps["linreg"].coef_, XTrain.columns))


def printEvalutation(X, Y, model):
    print("Mean squared error    : {:.5}".format(mean_squared_error(model.predict(X), Y)))
    print("Relative error        : {:.5%}".format(relative_error(model.predict(X), Y)))
    print("R-squared coefficient : {:.5}".format(model.score(X, Y)))


def slipDataset(X, Y):
    return train_test_split(X, Y, test_size=1/10, random_state=73)


#INIZIO DEL PROGRAMMA

#Da verificare il corretto utilizzo di datasheet.
#Esso può essere richiamato e utilizzato dalla funzione, senza l'obbligo di essere passato alle funzioni stesse come argomento.
#Può essere considerato con scope globale all'interno del progetto.
dataset = loadCSVFile(str(getRelativePath()) + str(FILEPATH))
dataset.set_index(["UserID", "ProductID"], inplace=True)

print(dataset["Age"].values)
dataset = binarizza(dataset)





'''ANALISI ESPLORATIVA'''
#exploratoryAnalysis(dataset)
dataElaboration(dataset)
