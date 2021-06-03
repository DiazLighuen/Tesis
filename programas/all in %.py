import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import datetime
import pandas
import json
import locale

locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 


# =============================================================================
# # google
# dfGoogle = pandas.read_csv('alphabet-google.csv').loc[:, 'Exchange Date':'Close']
# googleDate = dfGoogle['Exchange Date'].tolist()
# googleDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), googleDate))
# googleClose = dfGoogle['Close'].tolist()
# googleClose = list(map(locale.atof,googleClose))
# googleClose = list(map(lambda x : x/1054,googleClose))
# 
# # facebook
# dfFacebook = pandas.read_csv('facebook.csv').loc[:, 'Exchange Date':'Close']
# facebookDate = dfFacebook['Exchange Date'].tolist()
# facebookDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), facebookDate))
# facebookClose = dfFacebook['Close'].tolist()
# facebookClose = list(map(locale.atof,facebookClose))
# facebookClose = list(map(lambda x : x/135,facebookClose))
# =============================================================================

# gold
dfGold = pandas.read_csv('gold.csv').loc[:, 'Exchange Date':'Close']
goldDate = dfGold['Exchange Date'].tolist()
goldDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), goldDate))[::-1]
goldClose = dfGold['Close'].tolist()
goldClose = list(map(locale.atof,goldClose))[::-1]
#goldClose = list(map(lambda x : x/1284,goldClose))

# bitcoin
dfBitcoin = pandas.read_csv('bitcoin.csv').loc[:, 'time':'close']
bitcoinDate = dfBitcoin['time'].tolist()
bitcoinDate = list(map(lambda x : datetime.datetime.fromtimestamp(x).strftime('%d-%m-%Y'), bitcoinDate))
bitcoinClose = dfBitcoin['close'].tolist()
#bitcoinClose = list(map(locale.atof,bitcoinClose))
#bitcoinClose = list(map(lambda x : x/3835,bitcoinClose))
 
# =============================================================================
# # nasdaq100
# dfNasdaq100 = pandas.read_csv('nasdaq100.csv').loc[:, 'Exchange Date':'Close']
# nasdaq100Date = dfNasdaq100['Exchange Date'].tolist()
# nasdaq100Date = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), nasdaq100Date))
# nasdaq100Close = dfNasdaq100['Close'].tolist()
# nasdaq100Close = list(map(locale.atof,nasdaq100Close))
# nasdaq100Close = list(map(lambda x : x/6360,nasdaq100Close))
# 
# # nvidia
# dfNvidia = pandas.read_csv('nvidia.csv').loc[:, 'Exchange Date':'Close']
# nvidiaDate = dfNvidia['Exchange Date'].tolist()
# nvidiaDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), nvidiaDate))
# nvidiaClose = dfNvidia['Close'].tolist()
# nvidiaClose = list(map(locale.atof,nvidiaClose))
# nvidiaClose = list(map(lambda x : x/136,nvidiaClose))
# 
# # tesla
# dfTesla = pandas.read_csv('tesla.csv').loc[:, 'Exchange Date':'Close']
# teslaDate = dfTesla['Exchange Date'].tolist()
# teslaDate = list(map(lambda x : datetime.datetime.strptime(x,'%d-%b-%Y').strftime('%d-%m-%Y'), teslaDate))
# teslaClose = dfTesla['Close'].tolist()
# #teslaClose = list(map(locale.atof,teslaClose))
# teslaClose = list(map(lambda x : x/310,teslaClose))
# =============================================================================



#Grid configuration
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid((1,1), (0,0))
for label in ax1.xaxis.get_ticklabels():
     label.set_rotation(45)

ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)


#print (bitcoinDate)
#print (googleClose)
#print (facebookClose)
#print (goldClose)
#print (bitcoinClose)

#Plots
plt.plot( bitcoinDate, bitcoinClose, color='blue', label='Bitcoin' , linewidth=2)
plt.plot( goldDate, goldClose, color='blue', label='Gold' , linewidth=2)
#plt.plot( facebookDate, facebookClose, color='green', label='Facebook' , linewidth=2)
#plt.plot( googleDate, googleClose, color='red', label='Google' , linewidth=2)
#plt.plot( nasdaq100Date, nasdaq100Close, color='black', label='Nasdaq100' , linewidth=2)
#plt.plot( nvidiaDate, nvidiaClose, color='pink', label='Nvidia' , linewidth=2)
#plt.plot( teslaDate, teslaClose, color='brown', label='Tesla' , linewidth=2)

#Date adjust
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
#ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
#ax1.grid(True)

#Show
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Gold')#'Google vs Facebook vs Gold vs Bitcoin vs Nasdaq100 vs Nvidia vs Tesla')
plt.show()