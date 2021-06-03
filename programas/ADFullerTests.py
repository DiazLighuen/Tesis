import pandas
import statsmodels.tsa.stattools as ts

import locale

locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 

acceptPValue= 0.05
# =============================================================================
# Prices
# =============================================================================
print('Prices:')
print('')


#Bitcoin
dfBitcoin = pandas.read_csv('bitcoin.csv').loc[:, 'close']
bitcoinClose = dfBitcoin.tolist()
if(ts.adfuller(bitcoinClose)[1]<acceptPValue):
    print("Bitcoin is Stationary" ,ts.adfuller(bitcoinClose))
else:
    print("Bitcoin is Non-Stationary" ,ts.adfuller(bitcoinClose))
print('')

#Gold
dfGold = pandas.read_csv('gold.csv').loc[:, 'Close']
goldClose = dfGold.tolist()
goldClose = list(map(locale.atof,goldClose))[::-1]
if(ts.adfuller(goldClose)[1]<acceptPValue):
    print("Gold is Stationary" ,ts.adfuller(goldClose))
else:
    print("Gold is Non-Stationary" ,ts.adfuller(goldClose))
print('')


#Nvidia
dfNvidia = pandas.read_csv('nvidia.csv').loc[:,'Close']
nvidiaClose = dfNvidia.tolist()
if(ts.adfuller(bitcoinClose)[1]<acceptPValue):
    print("Nvidia is Stationary" ,ts.adfuller(bitcoinClose))
else:
    print("Nvidia is Non-Stationary" ,ts.adfuller(bitcoinClose))

print('')
print('')

# =============================================================================
# First Differences
# =============================================================================
print('First Differences:')
print('')

    
dfTraining = pandas.read_csv('TrainingSet.csv')
dfTest = pandas.read_csv('TestingSet.csv')

#Bitcoin
bitcoinFD = dfTraining['Bitcoin'].tolist()
bitcoinFD.extend(dfTest['Bitcoin'].tolist())
if(ts.adfuller(bitcoinFD)[1]<acceptPValue):
    print("Bitcoin is Stationary" ,ts.adfuller(bitcoinFD))
else:
    print("Bitcoin is Non-Stationary" ,ts.adfuller(bitcoinFD))
print('')

#Gold
goldFD = dfTraining['Gold'].dropna().tolist()
goldFD.extend(dfTest['Gold'].dropna().tolist())
if(ts.adfuller(goldFD)[1]<acceptPValue):
    print("Gold is Stationary" ,ts.adfuller(goldFD))
else:
    print("Gold is Non-Stationary" ,ts.adfuller(goldFD))
print('')

#Nvidia
nvidiaFD = dfTraining['Nvidia'].dropna().tolist()
nvidiaFD.extend(dfTest['Nvidia'].dropna().tolist())
if(ts.adfuller(nvidiaFD)[1]<acceptPValue):
    print("Nvidia is Stationary" ,ts.adfuller(nvidiaFD))
else:
    print("Nvidia is Non-Stationary" ,ts.adfuller(nvidiaFD))
    
print('')
print('')

# =============================================================================
# Residuals of ARMA
# =============================================================================
print('Residuals of ARMA:')
print('')

dfResiduals = pandas.read_csv('residuals.csv')

#Bitcoin
bitcoinFD = dfResiduals['Bitcoin'].tolist()
if(ts.adfuller(bitcoinFD)[1]<acceptPValue):
    print("Bitcoin is Stationary" ,ts.adfuller(bitcoinFD))
else:
    print("Bitcoin is Non-Stationary" ,ts.adfuller(bitcoinFD))
print('')

#Gold
goldFD = dfResiduals['Gold'].dropna().tolist()
if(ts.adfuller(goldFD)[1]<acceptPValue):
    print("Gold is Stationary" ,ts.adfuller(goldFD))
else:
    print("Gold is Non-Stationary" ,ts.adfuller(goldFD))
print('')

#Nvidia
nvidiaFD = dfResiduals['Nvidia'].dropna().tolist()
if(ts.adfuller(nvidiaFD)[1]<acceptPValue):
    print("Nvidia is Stationary" ,ts.adfuller(nvidiaFD))
else:
    print("Nvidia is Non-Stationary" ,ts.adfuller(nvidiaFD))
    
    
# =============================================================================
# Prices:
# 
# Bitcoin is Non-Stationary (-1.3596343587427853, 0.6014677457978088, 1, 406, {'1%': -3.4465596717208813, '5%': -2.8686852499495843, '10%': -2.570576203741901}, 5590.13539109512)
# 
# Gold is Non-Stationary (-0.683287555979917, 0.8510310770234317, 14, 276, {'1%': -3.4542672521624214, '5%': -2.87206958769775, '10%': -2.5723807881747534}, 2061.629194729552)
# 
# Nvidia is Non-Stationary (-1.3596343587427853, 0.6014677457978088, 1, 406, {'1%': -3.4465596717208813, '5%': -2.8686852499495843, '10%': -2.570576203741901}, 5590.13539109512)
# 
# 
# First Differences:
# 
# Bitcoin is Stationary (-21.461560181807748, 0.0, 0, 406, {'1%': -3.4465596717208813, '5%': -2.8686852499495843, '10%': -2.570576203741901}, -1482.863881987742)
# 
# Gold is Stationary (-3.78419091660936, 0.003073267924879881, 13, 276, {'1%': -3.4542672521624214, '5%': -2.87206958769775, '10%': -2.5723807881747534}, -1922.5858052364592)
# 
# Nvidia is Stationary (-17.063173905377397, 7.920097814217872e-30, 0, 280, {'1%': -3.453922368485787, '5%': -2.871918329081633, '10%': -2.5723001147959184}, -1249.6274445803979)
# 
# 
# Residuals of ARMA:
# 
# Bitcoin is Stationary (-20.100295585775456, 0.0, 0, 405, {'1%': -3.446599953548936, '5%': -2.86870295908671, '10%': -2.570585643956714}, -1478.298577368802)
# 
# Gold is Stationary (-16.595895353685478, 1.7820262006924515e-29, 0, 288, {'1%': -3.453261605529366, '5%': -2.87162848654246, '10%': -2.5721455328896603}, -1915.072530642859)
# 
# Nvidia is Stationary (-11.542799972788476, 3.60040111549817e-21, 1, 278, {'1%': -3.4540935579190495, '5%': -2.8719934111688965, '10%': -2.5723401594120388}, -1254.7393164015682)
# 
# =============================================================================
