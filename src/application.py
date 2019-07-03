import csv
import pandas as pd
import numpy as np
import os.path as fileSystem
import matplotlib.pyplot as plot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


FILEPATH = "day.csv"

'''La funzione crea in grafico a dispersione'''
#def showDispersionGraph(feature1, feature2):
#    import matplotlib.pyplot as plt
#    plt.scatter(feature1, feature2)
#    plt.show()

'''sei diversi metodi di elaborazione: regressione senza vincoli, Ridge e Lasso
producono diversi modelli di previsione'''

def elaborationWithLasso(degeePipe, alphaPipe):
    return Pipeline([("poly", PolynomialFeatures(degree=degeePipe, include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", Lasso(alpha=alphaPipe, tol=0.001))])

def ElaborationKFold(X, Y):
    kf = KFold(n_splits=5, shuffle=True, random_state=73)
    model = elaborationWithLasso(6, 8)
    scores = cross_val_score(model, X, Y, cv=kf)
    print(scores)

def elaborationWithPerceptron(XTrain, YTrain, dg):
    return Pipeline([("scaler",  StandardScaler()),
                    ("model",  Perceptron(penalty="l2", alpha=0.0005, max_iter=10))])

def elaborationWithRidge(degeePipe):
    return Pipeline([("poly", PolynomialFeatures(degree=degeePipe, include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", Ridge(alpha=5))])

def elaborationWithElasticNetdef(degeePipe=1):
    return Pipeline([("scale",  StandardScaler()),
                     ("regr",  ElasticNet(alpha=8, l1_ratio=1))])

def elaborationWithoutRestrain(degeePipe):
    return Pipeline([("poly",   PolynomialFeatures(degree=degeePipe, include_bias=False)),
                    ("scale",  StandardScaler()),
                    ("linreg", LinearRegression())])

def relativeError(YTrue, YPred):
    return np.mean(np.abs((YTrue - YPred) / YTrue))

def printEvalutation(X, Y, model):
    print("Mean squared error    : {:.5}".format(mean_squared_error(model.predict(X), Y)))
    print("Relative error        : {:.5%}".format(relativeError(model.predict(X), Y)))
    print("R-squared coefficient : {:.5}".format(model.score(X, Y)))

def testingElaboration(XTrain, YTrain, XVal, YVal):

    print("Lasso")
    model = elaborationWithLasso(6, 8)
    model.fit(XTrain, YTrain)
    printEvalutation(XVal, YVal, model)

    print("no Restain")
    model = elaborationWithoutRestrain(2)
    model.fit(XTrain, YTrain)
    printEvalutation(XVal, YVal, model)

    print("Net")
    ''' Capire se nella funzione va messo PolynomialFeatures'''
    model = elaborationWithElasticNetdef()
    model.fit(XTrain, YTrain)
    printEvalutation(XVal, YVal, model)

    print("Ridge")
    ''' Capire se nella funzione va messo PolynomialFeatures'''
    model = elaborationWithRidge(4)
    model.fit(XTrain, YTrain)
    printEvalutation(XVal, YVal, model)

    print("Perceptron")
    model = elaborationWithPerceptron(XTrain, YTrain, 4)
    model.fit(XTrain, YTrain)
    printEvalutation(XVal, YVal, model)

'''l'idea è quella di utilizare un modello a Lasso di primo grado per determinare
le variabili inutili, che vengono eliminate dal dataset.'''

def showZerosFeatures(XTrain, YTrain):
    model = elaborationWithLasso(1, 8)
    model.fit(XTrain, YTrain)
    tmp = pd.Series(model.named_steps["linreg"].coef_, XTrain.columns)
    print(tmp)
    a = []
    for row in tmp.index:
        if(tmp[row]==0):
            a.append(row)
    print(a)

def testingGridSerach(XTrain, YTrain, XVal, YVal):

    print("Lasso")
    parLasso = {
        "poly__degree": [2,6,8],
        "linreg__alpha":  [2,6,8]
    }
    model = Pipeline([("poly", PolynomialFeatures(include_bias=False)),
                    ("scale",  StandardScaler()),   # <- aggiunto
                    ("linreg", Lasso(tol=0.001))])
    gs = GridSearchCV(model, param_grid=parLasso)
    gs.fit(XTrain, YTrain)
    print(gs.best_params_)
    printEvalutation(XVal, YVal, gs)

#    print("no Restain")
#    parNR = {
#        "poly__degree": [1,2,6,8],
#    }
#    model = Pipeline([("poly",   PolynomialFeatures(include_bias=False)),
#                    ("scale",  StandardScaler()),
#                    ("linreg", LinearRegression())])
#    gs = GridSearchCV(model, param_grid=parNR)
#    gs.fit(XTrain, YTrain)
#    print(gs.best_params_)
#    printEvalutation(XVal, YVal, gs)
#
#    print("Net")
#    ''' Capire se nella funzione va messo PolynomialFeatures'''
#    parNet = {
#        "regr__alpha": [1,2,6,8],
#        "regr__l1_ratio": [0.1, 0.5, 1.0]
#    }
#    model = Pipeline([("scale",  StandardScaler()),
#                    ("regr",  ElasticNet())])
#    gs = GridSearchCV(model, param_grid=parNet)
#    gs.fit(XTrain, YTrain)
#    print(gs.best_params_)
#    printEvalutation(XVal, YVal, gs)
#
#    print("Ridge")
#    parRidge = {
#        "poly__degree": [1,2,6,8],
#        "linereg__alpha":  [1,2,6,8]
#    }
#    ''' Capire se nella funzione va messo PolynomialFeatures'''
#    model = Pipeline([("poly", PolynomialFeatures(include_bias=False)),
#                    ("scale",  StandardScaler()),   # <- aggiunto
#                    ("linreg", Ridge())])
#    gs = GridSearchCV(model, param_grid=par)
#    gs.fit(XTrain, YTrain)
#    print(gs.best_params_)
#    printEvalutation(XVal, YVal, gs)
#
#    '''non ho internet e non so quali sono i valori da dare ad alpha'''
#    #print("Perceptron")
#    #parNR = {
#    #    "scaler__alpha": [1,2,6,8],
#    #}
#    #Pipeline([("scaler",  StandardScaler()),
#    #        ("model",  Perceptron(penalty="l2", alpha=0.0005, max_iter=10))])
#    #gs = GridSearchCV(model, param_grid=par)
#    #gs.fit(XTrain, YTrain)
#    #print(gs.best_params_)
#    #printEvalutation(XVal, YVal, gs)



def slipDataset(X, Y):
    return train_test_split(X, Y, test_size=0.33, random_state=73)

def dataElaboration(dataFrame):
    Y = dataFrame["cnt"]
    X = dataFrame.drop(["casual", "registered", "cnt"], axis=1)
    XTrain, XVal, YTrain, YVal = slipDataset(X, Y)

    print("Vializzazione delle feature che vanno a 0 utilizzando il LASSO.")
    showZerosFeatures(XTrain, YTrain)

    print("Vengono visualizzati i risultati ottenuti con alcuni algoritmi per la generazioni di modelli di regressione.")
    testingGridSerach(XTrain, YTrain, XVal, YVal);

    print("Viene testata la Pipeline applicato il LASSO, poiche' ritenuta la migliore a livello prestazionale.")
    print("Il K-Fold permette di suddividere il set in n parti, aventi la stessa grandezza, e di compiere le operazioni di Train e Validation su di essere.")
    ElaborationKFold(X, Y)

def showHistogram(feature):
    feature.plot.hist(bins=20)
    plot.show()

def plotModelOnData(x, y, XAxisName, YAxisName, model=None):
    plot.scatter(x, y)
    if model is not None:
        xlim, ylim = plot.gca().get_xlim(), plot.gca().get_ylim()
        line_x = np.linspace(xlim[0], xlim[1], 100)
        line_y = model.predict(line_x[:, None])
        plot.plot(line_x, line_y, c="red", lw=3)
        plot.xlim(xlim); plt.ylim(ylim)
    plot.grid()
    plot.xlabel(XAxisName); plot.ylabel(YAxisName)
    plot.show()

'''
Restituisce l'indice di correlazione tra due feature.
param:
feature1, feature2: nparray or series
returns:
indice di correlazione
'''
def getCorrelation(feature1, feature2):
    return np.mean((feature1-feature1.mean()) * (feature2-feature2.mean())) / (feature1.std() * feature2.std())

def correlationRank(dataset, feature):
    correlation = []
    dataset = dataset.drop(["casual", "registered"], axis=1)
    for a in dataset.columns:
        print(a)
        correlation.append(getCorrelation(dataset[a].astype("float"), feature))
        #plotModelOnData(dataset[a].astype("float"), feature, a, "Byke Rent")
        #showHistogram(dataset[a].astype("int"))
    tmp = pd.Series(correlation, dataset.columns)
    tmp.sort_values(ascending=False, inplace=True)
    print( tmp)

'''La funzione visualizza il dtype di ogni ottributo del dataFrame passatogli.
Aggiunge infine anche l'occupazione in memoria.'''
def generalDataFrameInfo(dataFrame):
    dataFrame.info(memory_usage="deep")

'''La funzione describe tende ad escludere il primo attributo.
Probabilmente perchè di tipo object e quindi non ha competenze per il calcolo dei valori. '''
def exploratoryAnalysis(dataFrame):
    generalDataFrameInfo(dataFrame)
    print(dataFrame.describe())
    correlationRank(dataFrame.drop(["cnt"], axis=1), dataFrame["cnt"])

'''Zona del programma in cui vengono collocate le funzioni.
Esse verranno chiamate all'occorrenza all'interno del programma'''
def loadCSVFile(path):
    if fileSystem.exists(path):
        return pd.read_csv(path, sep=",")
    else:
        print("File non trovato")
        return None

def getRelativePath():
    return fileSystem.dirname(fileSystem.dirname(__file__)) + "\\res\\datasheet\\"

def createDataset():
    dataset = loadCSVFile(str(getRelativePath()) + str(FILEPATH))
    dataset.set_index(["dteday"], inplace=True)
    return dataset.drop(["instant"], axis=1)

def main():
    dataset = createDataset()
    exploratoryAnalysis(dataset)
    dataElaboration(dataset)

#INIZIO DEL PROGRAMMA
if(__name__ == "__main__"):
    main()
