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

#plot.scatter(ds["instant"], ds["cnt"])
#line_y = model.predict(X)
#plot.plot(ds["instant"], line_y, c="red", lw=3)
#plot.grid()
#plot.show()

'''sei diversi metodi di elaborazione: regressione senza vincoli, Ridge e Lasso
producono diversi modelli di previsione'''

def elaborationWithLasso(degeePipe=1, alphaPipe=0):
    return Pipeline([("poly", PolynomialFeatures(degree=degeePipe, include_bias=False)),
                    ("scale",  StandardScaler()),
                    ("linreg", Lasso(alpha=alphaPipe, max_iter=6000, tol=0.005))])

def elaborationWithRidge():
    return Pipeline([("poly", PolynomialFeatures(include_bias=False)),
                    ("scale",  StandardScaler()),
                    ("linreg", Ridge())])

def elaborationWithElasticNet():
    return Pipeline([("poly",   PolynomialFeatures(include_bias=False)),
                     ("scale",  StandardScaler()),
                     ("linreg",  ElasticNet(tol = 0.05, max_iter = 6000))])

def elaborationWithoutRestrain():
    return Pipeline([("poly",  PolynomialFeatures(include_bias=False)),
                    ("scale",  StandardScaler()),
                    ("linreg", LinearRegression())])

''' Funzioni per la valutazione dei modelli e delle feature'''

def relativeError(YTrue, YPred):
    return np.mean(np.abs((YTrue - YPred) / YTrue))

def printEvalutation(X, Y, model):
    print("Mean squared error    : {:.5}".format(mean_squared_error(model.predict(X), Y)))
    print("Relative error        : {:.5%}".format(relativeError(model.predict(X), Y)))
    print("R-squared coefficient : {:.5}".format(model.score(X, Y)))

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

def getCorrelation(feature1, feature2):
    return np.mean((feature1-feature1.mean()) * (feature2-feature2.mean())) / (feature1.std() * feature2.std())

def correlationRank(dataset, feature):
    correlation = []
    dataset = dataset.drop(["casual", "registered"], axis=1)
    for a in dataset.columns:
        print(a)
        correlation.append(getCorrelation(dataset[a].astype("float"), feature))
        plotModelOnData(dataset[a].astype("float"), feature, a, "Byke Rent")
    tmp = pd.Series(correlation, dataset.columns)
    tmp.sort_values(ascending=False, inplace=True)
    print( tmp)

def EvalutationTable(results):
    return pd.DataFrame(results.cv_results_).sort_values("mean_test_score", ascending=False)


'''Funzioni di elaborazione'''

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
    model = elaborationWithElasticNetdef()
    model.fit(XTrain, YTrain)
    printEvalutation(XVal, YVal, model)

    print("Ridge")
    model = elaborationWithRidge(4)
    model.fit(XTrain, YTrain)
    printEvalutation(XVal, YVal, model)


def ElaborationKFold(X, Y):
    kf = KFold(n_splits=5, shuffle=True, random_state=73)
    model = elaborationWithLasso(6, 8)
    scores = cross_val_score(model, X, Y, cv=kf)
    print(scores)


def testingGridSerach(XTrain, YTrain, XVal, YVal):
    print("Lasso")
    parLasso = {
        "poly__degree": [1,6,8],
        "linreg__alpha":  [1,5,8]
    }
    LassoModel = elaborationWithLasso()
    lassoGridSearch = GridSearchCV(LassoModel, param_grid=parLasso)
    lassoGridSearch.fit(XTrain, YTrain)
    print(lassoGridSearch.best_params_)
    EvalutationTable(lassoGridSearch)
    printEvalutation(XVal, YVal, lassoGridSearch)

    print("No Restain")
    parNR = {
       "poly__degree": [1,2],
    }
    NRmodel = elaborationWithoutRestrain()
    NRGridSearch = GridSearchCV(NRmodel, param_grid=parNR)
    NRGridSearch.fit(XTrain, YTrain)
    EvalutationTable(NRGridSearch)
    print(NRGridSearch.best_params_)

    print("ElasticNet")
    parNet = {
        "poly__degree": [1,2,6],
        "linreg__alpha": [1,2,8],
        "linreg__l1_ratio": [0.1, 0.5, 1.0]
    }
    NETmodel = elaborationWithElasticNet()
    ENgridSearch = GridSearchCV(NETmodel, param_grid=parNet)
    ENgridSearch.fit(XTrain, YTrain)
    print(ENgridSearch.best_params_)
    EvalutationTable(ENgridSearch)
    printEvalutation(XVal, YVal, ENgridSearch)


    print("Ridge")
    parRidge = {
        "poly__degree": [1,6,8],
        "linreg__alpha":  [1,2,6]
    }
    model = elaborationWithRidge()
    ridgeGridSearch = GridSearchCV(model, param_grid=parRidge)
    ridgeGridSearch.fit(XTrain, YTrain)
    print(ridgeGridSearch.best_params_)
    EvalutationTable(ridgeGridSearch)
    printEvalutation(XVal, YVal, ridgeGridSearch)


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

'''funzioni di analisi e mdellazione del dataset'''

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

def slipDataset(X, Y):
    return train_test_split(X, Y, test_size=0.33, random_state=73)

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

'''main'''

def main():
    dataset = createDataset()
    exploratoryAnalysis(dataset)
    dataElaboration(dataset)

#INIZIO DEL PROGRAMMA
if(__name__ == "__main__"):
    main()
