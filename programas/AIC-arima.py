import pandas
from statsmodels.tsa import stattools



df = pandas.read_csv('TrainingSet.csv')
date = df['Date'].tolist()
bitcoinFD = df['Bitcoin'].tolist()
goldFD = df['Gold'].dropna().tolist()
NvidiaFD = df['Nvidia'].dropna().tolist()

# =============================================================================
# ic = 'aic' - Akaike Information Criterion options=('aic', 'bic', 't-stat')
# 
# trend = 'c'
# 'c' : constant only (default).
# 'ct' : constant and trend.
# 'ctt' : constant, and linear and quadratic trend.
# 'nc' : no constant, no trend.
# =============================================================================
        

armaModelBitcoin = stattools.arma_order_select_ic(bitcoinFD, max_ar=4, max_ma=4, ic='aic', trend='c')
armaModelGold = stattools.arma_order_select_ic(goldFD, max_ar=4, max_ma=4, ic='aic', trend='c')
armaModelNvidia = stattools.arma_order_select_ic(NvidiaFD, max_ar=4, max_ma=4, ic='aic', trend='c')

print (armaModelBitcoin)
print (armaModelGold)
print (armaModelNvidia)

# =============================================================================
# Bitcoin
# =============================================================================
# =============================================================================
# primeras-diferencias
# {'aic':
#    0            1            2            3            4
# 0              -1556.041101 -1554.118219 -1552.399971 -1550.589078
# 1 -1556.073765 -1554.081441 -1552.198847 -1550.763837 -1548.781882
# 2 -1554.088495 -1552.161373 -1553.298577 -1556.447533 -1546.780347
# 3 -1552.428002 -1550.843304 -1556.488245 -1549.420889 -1547.463026
# 4 -1550.514585 -1548.879764 -1546.884735 -1550.550670 -1555.790540, 'aic_min_order': (3, 2)}
# =============================================================================

# =============================================================================
# TrainingTest
# {'aic':        
#    0            1            2            3            4
# 0              -1054.915540 -1053.171810 -1052.092096 -1050.442042
# 1 -1054.997638 -1053.015224 -1051.564151 -1051.117352 -1049.142307
# 2 -1053.042545 -1051.403307 -1056.394710 -1057.599621 -1058.607543
# 3 -1052.208607 -1051.495047 -1031.900845 -1054.958669 -1053.887374
# 4 -1050.401597 -1049.556054 -1047.501966 -1051.462238 -1053.836218, 'aic_min_order': (2, 4)}
# =============================================================================



# =============================================================================
# Gold
# =============================================================================
# =============================================================================
# primeras-diferencias
# {'aic':
#    0            1            2            3            4
# 0              -2043.055815 -2041.223821 -2039.547770 -2038.886453
# 1 -2043.057477 -2043.666770 -2041.772652 -2039.791277 -2037.270228
# 2 -2041.213446 -2041.770989 -2039.926516 -2038.491316 -2042.923010
# 3 -2039.598352 -2038.766329 -2038.541628 -2041.892236 -2039.943921
# 4 -2038.585565 -2036.977266 -2042.806845 -2036.739124 -2041.147575, 'aic_min_order': (1, 1)}
# =============================================================================

# =============================================================================
# TrainingTest
# {'aic':       
#    0            1            2            3            4
# 0              -1389.778000 -1387.955049 -1385.955695 -1385.159271
# 1 -1389.780929 -1390.917992 -1389.001623 -1387.039475 -1383.859625
# 2 -1387.930207 -1388.999682 -1388.017212 -1385.459187 -1387.165242
# 3 -1385.932775 -1387.022693 -1385.478208 -1389.438922 -1387.585106
# 4 -1384.900100 -1383.527997 -1387.154285 -1387.575292 -1386.790969, 'aic_min_order': (1, 1)}
# =============================================================================



# =============================================================================
# Nvidia
# =============================================================================
# =============================================================================
# primeras-diferencias
# {'aic':
#    0            1            2            3            4
# 0              -1268.589183 -1269.780337 -1268.391506 -1269.137463
# 1 -1268.578363 -1266.664185 -1267.977839 -1267.758137 -1267.326964
# 2 -1270.275711 -1268.470932 -1274.537078 -1273.031448 -1266.626332
# 3 -1268.748751 -1267.963134 -1268.465715 -1266.625690 -1264.636934
# 4 -1269.022871 -1267.368805 -1275.709221 -1273.709662 -1262.640066, 'aic_min_order': (4, 2)}
# =============================================================================

# =============================================================================
# TrainingTest
# {'aic':           
#    0           1           2           3           4
# 0             -846.938566 -848.152250 -846.695061 -846.883152
# 1 -846.918039 -845.524820 -846.346729 -845.854124 -845.168073
# 2 -848.798158 -846.885406 -851.840531 -852.302037 -850.754496
# 3 -846.981926 -845.795409 -846.061836 -852.081026 -851.575970
# 4 -846.393105 -844.822060 -853.160454 -848.953138 -849.804653, 'aic_min_order': (4, 2)}
# =============================================================================