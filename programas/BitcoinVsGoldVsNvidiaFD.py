import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas
import numpy as np
import datetime
import locale

locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 

# bitcoin
dfBitcoin = pandas.read_csv('bitcoin.csv').loc[:, 'time':'close']
bitcoinDate = dfBitcoin['time'].tolist()
bitcoinDate = list(map(lambda x : datetime.datetime.utcfromtimestamp(x).strftime('%d-%m-%Y'), bitcoinDate))
bitcoinClose = dfBitcoin['close'].tolist()
bitcoinFD = np.diff(list(map(lambda x : np.log(x),bitcoinClose)))
bitcoinDate = bitcoinDate[1:]

# gold
dfGold = pandas.read_csv('gold.csv').loc[:, 'Exchange Date':'Close']
goldDate = dfGold['Exchange Date'].tolist()
goldDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), goldDate))[::-1]
goldClose = dfGold['Close'].tolist()
goldClose = list(map(locale.atof,goldClose))[::-1]
goldFD = np.diff(list(map(lambda x : np.log(x),goldClose)))
goldDate = goldDate[1:]


# nvidia
dfNvidia = pandas.read_csv('nvidia.csv').loc[:, 'Exchange Date':'Close']
nvidiaDate = dfNvidia['Exchange Date'].tolist()
nvidiaDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), nvidiaDate))
nvidiaClose = dfNvidia['Close'].tolist()
nvidiaFD = np.diff(list(map(lambda x : np.log(x),nvidiaClose)))
nvidiaDate = nvidiaDate[1:]

#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)

#Plots
#plt.plot( bitcoinDate, bitcoinFD, color='blue', label='Bitcoin' , linewidth=2)
#plt.plot( goldDate, goldFD, color='blue', label='Gold' , linewidth=2)
plt.plot( nvidiaDate, nvidiaFD, color='blue', label='Nvidia' , linewidth=2)

#Show
plt.xlabel('Date')
plt.ylabel('First differences')
plt.title('Nvidia')#'Bitcoin vs Gold vs Nvidia')
plt.show()

# =============================================================================
# #Create unique CSV
# fd = {}
# for x in range(len(bitcoinDate)):
#     fd[bitcoinDate[x]] = [bitcoinDate[x],bitcoinFD[x]]
# for x in range(len(goldDate)):
#     fd[goldDate[x]].append(goldFD[x])
# for x in range(len(nvidiaDate)):
#     fd[nvidiaDate[x]].append(nvidiaFD[x])
# 
# dfFd = pd.DataFrame(list(fd.values()), columns = ["Date","Bitcoin","Gold","Nvidia"])
# dfFd.to_csv("primeras-diferencias.csv", index=False)
# =============================================================================
