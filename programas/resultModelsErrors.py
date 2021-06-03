import pandas
import numpy as np


def index_agreement(s, o):
    """
	index of agreement
	
	input:
        s: list of simulated
        o: list of observed
    output:
        ia: index of agreement
    """
    ia = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
    return ia

def theil_inequality_coefficient(s, o):
    """
	theil inequality coefficient
	
	input:
        s: list of simulated
        o: list of observed
    output:
        tic: theil inequality coefficient
    """
    tic = (np.sqrt(np.mean((o-s)**2))/((np.sqrt(np.mean(np.square(o))))+(np.sqrt(np.mean(np.square(s))))))
    return tic


#Bitcoin
dfBitcoin = pandas.read_csv('bitcoinOutputARMA.csv')

bitcoinTestingSet = dfBitcoin['TestingSet']
bitcoinPredictions = dfBitcoin['Predictions']
bitcoinError = dfBitcoin['Error']
bitcoinSquareError = dfBitcoin['SquareError']
bitcoinAbsoluteError = dfBitcoin['AbsoluteError']

bitcoinARMAResult = ['ARMA',bitcoinSquareError.mean(),bitcoinAbsoluteError.mean(),index_agreement(bitcoinPredictions,bitcoinTestingSet),theil_inequality_coefficient(bitcoinPredictions,bitcoinTestingSet)]
dfFd = pandas.DataFrame([bitcoinARMAResult], columns = ["Model","RMSE","MAE","IA","TIC"])
dfFd.to_csv("bitcoinResult.csv", index=False)


#Gold
dfGold = pandas.read_csv('goldOutputARMA.csv')

goldTestingSet = dfGold['TestingSet']
goldPredictions = dfGold['Predictions']
goldError = dfGold['Error']
goldSquareError = dfGold['SquareError']
goldAbsoluteError = dfGold['AbsoluteError']

goldARMAResult = ['ARMA',goldSquareError.mean(),goldAbsoluteError.mean(),index_agreement(goldPredictions,goldTestingSet),theil_inequality_coefficient(goldPredictions,goldTestingSet)]

dfFd = pandas.DataFrame([goldARMAResult], columns = ["Model","RMSE","MAE","IA","TIC"])
dfFd.to_csv("goldResult.csv", index=False)

#Nvidia
dfNvidia = pandas.read_csv('nvidiaOutputARMA.csv')

nvidiaTestingSet = dfNvidia['TestingSet']
nvidiaPredictions = dfNvidia['Predictions']
nvidiaError = dfNvidia['Error']
nvidiaSquareError = dfNvidia['SquareError']
nvidiaAbsoluteError = dfNvidia['AbsoluteError']

nvidiaARMAResult = ['ARMA',nvidiaSquareError.mean(),nvidiaAbsoluteError.mean(),index_agreement(nvidiaPredictions,nvidiaTestingSet),theil_inequality_coefficient(nvidiaPredictions,nvidiaTestingSet)]

dfFd = pandas.DataFrame([nvidiaARMAResult], columns = ["Model","RMSE","MAE","IA","TIC"])
dfFd.to_csv("nvidiaResult.csv", index=False)
