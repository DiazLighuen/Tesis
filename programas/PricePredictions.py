import pandas
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
import numpy as np
import datetime
import locale

locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 


#Date and TestingSet
dfTest = pandas.read_csv('TestingSet.csv')
dfTraining = pandas.read_csv('TrainingSet.csv')

#Bitcoin
dfBitcoinOutput = pandas.read_csv('bitcoinOutputARMA.csv')

# =============================================================================
# bitcoinDateTest = dfTest['Date'].tolist()
# bitcoinDateTraining = dfTraining['Date'].tolist()
# bitcoinFDTraining = dfTraining['Bitcoin'].tolist()
# bitcoinFDTest = dfTest['Bitcoin'].tolist()
# =============================================================================

dfBitcoin = pandas.read_csv('bitcoin.csv').loc[:, 'time':'close']
bitcoinDate = dfBitcoin['time'].tolist()
bitcoinDate = list(map(lambda x : datetime.datetime.fromtimestamp(x).strftime('%d-%m-%Y'), bitcoinDate))
bitcoinClose = dfBitcoin['close'].tolist()
bitcoinDateTest = dfTest['Date'].tolist()

bitcoinLastPrice = 8296.34
bitcoinFDTest = dfTest['Bitcoin'].tolist()
bitcoinPredictions = dfBitcoinOutput['Predictions']
bitcoinPrice = list(map(lambda x : (math.exp(x))*bitcoinLastPrice, np.cumsum(bitcoinPredictions)))
print(bitcoinPrice)


#Gold
dfGoldOutput = pandas.read_csv('goldOutputARMA.csv')

dfGold = pandas.read_csv('gold.csv').loc[:, 'Exchange Date':'Close']
goldDate = dfGold['Exchange Date'].tolist()
goldDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), goldDate))[::-1]
goldClose = dfGold['Close'].tolist()
goldClose = list(map(locale.atof,goldClose))[::-1]
goldDateTest = dfTest.loc[:, 'Date':'Gold'].dropna()['Date'].tolist()

goldLastPrice = 1489.45
goldFDTest = dfTest['Gold'].dropna().tolist()
goldPredictions = dfGoldOutput['Predictions']
goldPrice = list(map(lambda x : (math.exp(x))*goldLastPrice, np.cumsum(goldPredictions)))
print(goldPrice)


#Nvidia
dfNvidiaOutput = pandas.read_csv('nvidiaOutputARMA.csv')

dfNvidia = pandas.read_csv('nvidia.csv').loc[:, 'Exchange Date':'Close']
nvidiaDate = dfNvidia['Exchange Date'].tolist()
nvidiaDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), nvidiaDate))
nvidiaClose = dfNvidia['Close'].tolist()
#nvidiaClose = list(map(locale.atof,nvidiaClose))
nvidiaDateTest = dfTest.loc[:, 'Date':'Nvidia'].dropna()['Date'].tolist()

nvidiaLastPrice = 185.99
nvidiaTestingSet = dfTest['Nvidia'].dropna().tolist()
nvidiaPredictions = dfNvidiaOutput['Predictions']
nvidiaPrice = list(map(lambda x : (math.exp(x))*nvidiaLastPrice, np.cumsum(nvidiaPredictions)))
print(nvidiaPrice)


#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)

#Show bitcoin
plt.title('Bitcoin')
plt.xlabel('Date')
plt.ylabel('Price prediction')
plt.plot(bitcoinDate, bitcoinClose, color ='green')
plt.plot(bitcoinDateTest, bitcoinPrice, color='red')
plt.show()

#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)

#Show gold
plt.title('Gold')
plt.xlabel('Date')
plt.ylabel('Price prediction')
plt.plot(goldDate, goldClose, color ='green')
plt.plot(goldDateTest, goldPrice, color='red')
plt.show()

#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)

#Show gold
plt.title('Nvidia')
plt.xlabel('Date')
plt.ylabel('Price prediction')
plt.plot(nvidiaDate, nvidiaClose, color ='green')
plt.plot(nvidiaDateTest, nvidiaPrice, color='red')
plt.show()