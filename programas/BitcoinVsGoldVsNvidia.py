import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import datetime
import pandas
import json
import locale

locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 

# bitcoin
dfBitcoin = pandas.read_csv('bitcoin.csv').loc[:, 'time':'close']
bitcoinDate = dfBitcoin['time'].tolist()
bitcoinDate = list(map(lambda x : datetime.datetime.fromtimestamp(x).strftime('%d-%m-%Y'), bitcoinDate))
bitcoinClose = dfBitcoin['close'].tolist()
#bitcoinClose = list(map(lambda x : x/3835,bitcoinClose))
bitcoinFD = np.diff(list(map(lambda x : np.log(x),bitcoinClose)))

# gold
dfGold = pandas.read_csv('gold.csv').loc[:, 'Exchange Date':'Close']
goldDate = dfGold['Exchange Date'].tolist()
goldDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), goldDate))[::-1]
goldClose = dfGold['Close'].tolist()
goldClose = list(map(locale.atof,goldClose))[::-1]
#goldClose = list(map(lambda x : x/1284,goldClose))
goldFD = np.diff(list(map(lambda x : np.log(x),goldClose)))


# nvidia
dfNvidia = pandas.read_csv('nvidia.csv').loc[:, 'Exchange Date':'Close']
nvidiaDate = dfNvidia['Exchange Date'].tolist()
nvidiaDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), nvidiaDate))
nvidiaClose = dfNvidia['Close'].tolist()
#nvidiaClose = list(map(lambda x : x/136,nvidiaClose))
nvidiaFD = np.diff(list(map(lambda x : np.log(x),nvidiaClose)))

#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)

#Plots
plt.plot( bitcoinDate[1:], bitcoinFD, color='blue', label='Bitcoin' , linewidth=2)
plt.plot( goldDate[1:], goldFD, color='red', label='Gold' , linewidth=2)
plt.plot( nvidiaDate[1:], nvidiaFD, color='green', label='Nvidia' , linewidth=2)

#Show
plt.xlabel('Date')
plt.ylabel('First differences')
plt.title('Bitcoin vs Gold vs Nvidia')
plt.show()