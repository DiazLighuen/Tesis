import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error


dfTraining = pandas.read_csv('TrainingSet.csv')
bitcoinDateTraining = dfTraining['Date'].tolist()
bitcoinFDTraining = dfTraining['Bitcoin'].tolist()
goldDateTraining = dfTraining.loc[:, 'Date':'Gold'].dropna()['Date'].tolist()
goldFDTraining = dfTraining['Gold'].dropna().tolist()
nvidiaDateTraining = dfTraining.loc[:, 'Date':'Nvidia'].dropna()['Date'].tolist()
nvidiaFDTraining = dfTraining['Nvidia'].dropna().tolist()

dfTest = pandas.read_csv('TestingSet.csv')
bitcoinDateTest = dfTest['Date'].tolist()
bitcoinFDTest = dfTest['Bitcoin'].tolist()
goldDateTest = dfTest.loc[:, 'Date':'Gold'].dropna()['Date'].tolist()
goldFDTest = dfTest['Gold'].dropna().tolist()
nvidiaDateTest = dfTest.loc[:, 'Date':'Nvidia'].dropna()['Date'].tolist()
nvidiaFDTest = dfTest['Nvidia'].dropna().tolist()

residual= []

#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)

#Bitcoin
historyBitcoin = dfTraining['Bitcoin'].tolist()
predictionsBitcoin = list()
outputBitcoin = []

for t in range(len(bitcoinFDTest)):
    preOutputBitcoin = []

    model = ARMA(historyBitcoin, order=(2,4))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictionsBitcoin.append(yhat)
    obs = bitcoinFDTest[t]
    historyBitcoin.append(obs)
    preOutputBitcoin.append(obs)
    preOutputBitcoin.append(yhat[0])
    preOutputBitcoin.append(obs-yhat[0])
    preOutputBitcoin.append((obs-yhat[0])**2)
    preOutputBitcoin.append(abs(obs-yhat[0]))
    outputBitcoin.append(preOutputBitcoin)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(bitcoinFDTest, predictionsBitcoin)
print('Test MSE: %.3f' % error)
residual.append(model_fit.resid)

#Show
plt.title('Bitcoin')
plt.xlabel('Date')
plt.ylabel('First differences')
plt.plot(bitcoinDateTraining, bitcoinFDTraining, color ='green')
plt.plot(bitcoinDateTest, bitcoinFDTest, color='blue')
plt.plot(bitcoinDateTest, predictionsBitcoin, color='red')
plt.show()

dfFd = pandas.DataFrame(outputBitcoin, columns = ["TestingSet","Predictions","Error","SquareError","AbsoluteError"])
dfFd.to_csv("bitcoinOutputARMA.csv", index=False)


#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)

#Gold
historyGold = dfTraining['Gold'].dropna().tolist()
predictionsGold = list()
outputGold = []

for t in range(len(goldFDTest)):
    preOutputGold = []

    model = ARMA(historyGold, order=(1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictionsGold.append(yhat)
    obs = goldFDTest[t]
    historyGold.append(obs)
    preOutputGold.append(obs)
    preOutputGold.append(yhat[0])
    preOutputGold.append(obs-yhat[0])
    preOutputGold.append((obs-yhat[0])**2)
    preOutputGold.append(abs(obs-yhat[0]))
    outputGold.append(preOutputGold)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(goldFDTest, predictionsGold)
print('Test MSE: %.3f' % error)
residual.append(model_fit.resid)

#Show
plt.title('Gold')
plt.xlabel('Date')
plt.ylabel('First differences')
plt.plot(goldDateTraining, goldFDTraining, color ='green')
plt.plot(goldDateTest, goldFDTest, color='blue')
plt.plot(goldDateTest, predictionsGold, color='red')
plt.show()

dfFd = pandas.DataFrame(outputGold, columns = ["TestingSet","Predictions","Error","SquareError","AbsoluteError"])
dfFd.to_csv("goldOutputARMA.csv", index=False)


#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)

#Nvidia
historyNvidia = dfTraining['Nvidia'].dropna().tolist()
predictionsNvidia = list()
outputNvidia = []

for t in range(len(nvidiaFDTest)):
    preOutputNvidia = []

    model = ARMA(historyNvidia, order=(4,2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictionsNvidia.append(yhat)
    obs = nvidiaFDTest[t]
    historyNvidia.append(obs)
    preOutputNvidia.append(obs)
    preOutputNvidia.append(yhat[0])
    preOutputNvidia.append(obs-yhat[0])
    preOutputNvidia.append((obs-yhat[0])**2)
    preOutputNvidia.append(abs(obs-yhat[0]))
    outputNvidia.append(preOutputNvidia)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(nvidiaFDTest, predictionsNvidia)
print('Test MSE: %.3f' % error)
residual.append(model_fit.resid)


#Show
plt.title('Nvidia')
plt.xlabel('Date')
plt.ylabel('First differences')
plt.plot(nvidiaDateTraining, nvidiaFDTraining, color ='green')
plt.plot(nvidiaDateTest, nvidiaFDTest, color='blue')
plt.plot(nvidiaDateTest, predictionsNvidia, color='red')
plt.show()

dfFd = pandas.DataFrame(outputNvidia, columns = ["TestingSet","Predictions","Error","SquareError","AbsoluteError"])
dfFd.to_csv("nvidiaOutputARMA.csv", index=False)

d = dict( Bitcoin = pandas.array(residual[0]), Gold = pandas.array(residual[1]), Nvidia = pandas.array(residual[2]))
dfFd = pandas.DataFrame(dict([ (k,pandas.Series(v)) for k,v in d.items() ]))
dfFd.to_csv("residuals.csv", index=False)