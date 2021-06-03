import pandas
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot


dfTraining = pandas.read_csv('TrainingSet.csv')
dateTraining = dfTraining['Date'].tolist()
bitcoinFDTraining = dfTraining['Bitcoin'].tolist()
goldFDTraining = dfTraining['Gold'].dropna().tolist()
NvidiaFDTraining = dfTraining['Nvidia'].dropna().tolist()

dfTest = pandas.read_csv('TestingSet.csv')
dateTest = dfTest['Date'].tolist()
bitcoinFDTest = dfTest['Bitcoin'].tolist()
goldFDTest = dfTest['Gold'].dropna().tolist()
NvidiaFDTest = dfTest['Nvidia'].dropna().tolist()




#Bitcoin
armaBitcoin = ARMA(bitcoinFDTraining,order=(2,4))
armaBitcoinFit = armaBitcoin.fit()
print(armaBitcoinFit.summary())

# plot residual errors bitcoin
residualsBitcoin = pandas.DataFrame(armaBitcoinFit.resid)
residualsBitcoin.plot()
pyplot.show()
print(residualsBitcoin.describe())

# =============================================================================
# predictions = list()
# for t in range(len(bitcoinFDTest)):
# 	model = ARMA(history, order=(2,4))
# 	model_fit = model.fit(disp=0)
# 	output = model_fit.forecast()
# 	yhat = output[0]
# 	predictions.append(yhat)
# 	obs = test[t]
# 	history.append(obs)
# 	print('predicted=%f, expected=%f' % (yhat, obs))
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)
# # plot
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()
# 
# =============================================================================





#Gold
armaGold = ARMA(goldFDTraining,order=(0,0))
armaGoldFit = armaGold.fit()
print(armaGoldFit.summary())

# plot residual errors gold
residualsGold = pandas.DataFrame(armaGoldFit.resid)
residualsGold.plot()
pyplot.show()
print(residualsGold.describe())





#Nvidia
armaNvidia = ARMA(NvidiaFDTraining,order=(4,2))
armaNvidiaFit = armaNvidia.fit()
print(armaNvidiaFit.summary())

# plot residual errors nvidia
residualsNvidia = pandas.DataFrame(armaNvidiaFit.resid)
residualsNvidia.plot()
pyplot.show()
print(residualsNvidia.describe())