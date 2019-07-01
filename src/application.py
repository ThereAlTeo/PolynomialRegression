import csv
import pandas as pd
import numpy as np
import os.path as fileSystem
import matplotlib.pyplot as plot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, LogisticRegression, Perceptron
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
    return dataset.drop(["CityCategory", "Gender", "Age"], axis=1)

def getRelativePath():
    return fileSystem.dirname(fileSystem.dirname(__file__)) + "\\res\\datasheet\\"

'''La funzione describe tende ad escludere il primo attributo.
Probabilmente perchè di tipo object e quindi non ha competenze per il calcolo dei valori.'''
def exploratoryAnalysis(dataFrame):
    generalDataFrameInfo(dataFrame)
    print(dataFrame.describe())
    pd.cut(dataFrame["MaritalStatus"], 2).value_counts().plot.pie()
    plot.show()

'''tre diversi metodi di elaborazione: regressione senza vincoli, Ridge e Lasso
producono diversi modelli di previsione'''
def elaborationWithRidge(XTrain, YTrain, dg):
    prm = Pipeline([#("poly",   PolynomialFeatures(degree=dg, include_bias=False)), #se viene fatta di terzo grado non basta la memoria
                    ("scaler",  StandardScaler()),   # <- aggiunto , n_jobs=-1  
                    ("model",  Perceptron(penalty="l2", alpha=0.005, max_iter=10))
                    ])
    prm.fit(XTrain, YTrain)
    return prm

def elaborationWithElasticNetdef(XTrain, YTrain, dg):
    prm = Pipeline([#("poly",   PolynomialFeatures(degree=dg, include_bias=False)), #se viene fatta di terzo grado non basta la memoria
                     ("scale",  StandardScaler()),
                     ("regr",  KernelRidge(alpha=10, kernel="poly", degree=1))])
    prm.fit(XTrain, YTrain)
    return prm

def elaborationWithoutRestrain(XTrain, YTrain, dg):
    prm = Pipeline([ ("poly",   PolynomialFeatures(degree=dg, include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", LinearRegression())])
    prm.fit(XTrain, YTrain)
    return prm

def elaborationWithLasso(XTrain, YTrain, dg):
    model = Pipeline([("poly", PolynomialFeatures(degree=dg, include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", Lasso(alpha=2))])
    model.fit(XTrain, YTrain)
    return model

def plot_model_on_data(x, y, model=None):
    plot.scatter(x, y)
    if model is not None:
        xlim, ylim = plot.gca().get_xlim(), plot.gca().get_ylim()
        line_x = np.linspace(xlim[0], xlim[1], 100)
        line_y = model.predict(line_x[:, None])
        plot.plot(line_x, line_y, c="red", lw=3)
        plot.xlim(xlim); plt.ylim(ylim)
    plot.grid()
    plot.xlabel("Temperatura (°C)"); plt.ylabel("Consumi (GW)")
    plot.show()

def relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

'''prototipo: l'idea è quella di utilizare un modello a Lasso di primo grado per determinare
le variabili inutili, che vengono eliminate dal dataset.
viene creato successivamente un'altro modello dal dataset modificato
PS: migliora la previsione dal 36 al 34 % di errore ma non riesco a farlo fare in maniera automatica
Restituisce in modo da rendere omogeneo il modello con i dati di test'''
def multipleElaboration(XTrain, YTrain):
    model = elaborationWithLasso(XTrain, YTrain, 1)
    tmp = pd.Series(model.named_steps["linreg"].coef_, XTrain.columns)
    print("....................................")
    a = []
    for row in tmp.index:
        if(tmp[row]==0):
            a.append(row)
    XTrain = XTrain.drop(a, axis=1)
    model = elaborationWithRidge(XTrain, YTrain, 1)
    return model, a



def dataElaboration(dataFrame):
    Y = dataFrame["Purchase"].values
    X = dataFrame.drop(["ProductCategory3", "ProductCategory2", "Purchase"], axis=1)
    XTrain, XVal, YTrain, YVal = slipDataset(X, Y)
    #p = elaborationWithRidge(XTrain, YTrain, 2)
    #print(p)
    p, a = multipleElaboration(XTrain, YTrain)
    #print(p.named_steps["linreg"].coef_)
    print(XTrain.columns)
    XVal = XVal.drop(a, axis=1)
    printEvalutation(XVal, YVal, p)
    #plot_model_on_data(XVal, YVal, p)
    '''
    Non funziona ancora

    plot.scatter(XVal["ProductCategory"], YVal),
    line_x = XVal["ProductCategory1"]
    line_y = p.predict(XVal);
    plt.plot(line_x, line_y, c="red", lw=3)
    '''



def printEvalutation(X, Y, model):
    print("Mean squared error    : {:.5}".format(mean_squared_error(model.predict(X), Y)))
    print("Relative error        : {:.5%}".format(relative_error(model.predict(X), Y)))
    print("R-squared coefficient : {:.5}".format(model.score(X, Y)))


def slipDataset(X, Y):
    return train_test_split(X, Y, test_size=0.33, random_state=73)

def main():
    dataset = loadCSVFile(str(getRelativePath()) + str(FILEPATH))
    dataset.set_index(["UserID", "ProductID"], inplace=True)
    dataset = binarizza(dataset)
    #print(dataset)
    #exploratoryAnalysis(dataset)
    dataElaboration(dataset)

#INIZIO DEL PROGRAMMA
if(__name__ == "__main__"):
    main()
