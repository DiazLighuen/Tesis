import pyflux as pf
import pandas


df = pandas.read_csv('TrainingSet.csv')
bitcoinFD = df['Bitcoin'].to_numpy()
goldFD = df['Gold'].dropna().to_numpy()
NvidiaFD = df['Nvidia'].dropna().to_numpy()

# Bitcoin
# =============================================================================
model = pf.GARCH(bitcoinFD,p=0,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(bitcoinFD,p=1,q=0)
x = model.fit()
x.summary()

model = pf.GARCH(bitcoinFD,p=1,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(bitcoinFD,p=1,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(bitcoinFD,p=2,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(bitcoinFD,p=2,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(bitcoinFD,p=2,q=3)
x = model.fit()
x.summary()

model = pf.GARCH(bitcoinFD,p=3,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(bitcoinFD,p=3,q=3)
x = model.fit()
x.summary()

print("gold")
# Gold
# =============================================================================
model = pf.GARCH(goldFD,p=0,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(goldFD,p=1,q=0)
x = model.fit()
x.summary()

model = pf.GARCH(goldFD,p=1,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(goldFD,p=1,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(goldFD,p=2,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(goldFD,p=2,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(goldFD,p=2,q=3)
x = model.fit()
x.summary()

model = pf.GARCH(goldFD,p=3,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(goldFD,p=3,q=3)
x = model.fit()
x.summary()

print("nvidia")
# Nvidia
# =============================================================================
model = pf.GARCH(NvidiaFD,p=0,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(NvidiaFD,p=1,q=0)
x = model.fit()
x.summary()

model = pf.GARCH(NvidiaFD,p=1,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(NvidiaFD,p=1,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(NvidiaFD,p=2,q=1)
x = model.fit()
x.summary()

model = pf.GARCH(NvidiaFD,p=2,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(NvidiaFD,p=2,q=3)
x = model.fit()
x.summary()

model = pf.GARCH(NvidiaFD,p=3,q=2)
x = model.fit()
x.summary()

model = pf.GARCH(NvidiaFD,p=3,q=3)
x = model.fit()
x.summary()

#TrainingSet
# Bitcoin
# =============================================================================
# GARCH(0,1) AIC: -1064.0221                                   
# GARCH(1,0) AIC: -1049.0939                                   
# GARCH(1,1) AIC: -1070.7533                   <===============================                                   
# GARCH(1,2) AIC: -1063.4973                                   
# GARCH(2,1) AIC: -1063.6079                                   
# GARCH(2,2) AIC: -1061.5108                                   
# GARCH(2,3) AIC: -1054.516                                    
# GARCH(3,2) AIC: -1054.5869                                   
# GARCH(3,3) AIC: -1052.5354                                   
# =============================================================================

# Gold
# =============================================================================
# GARCH(0,1) AIC: -1383.2254                                   
# GARCH(1,0) AIC: -1382.4525                                   
# GARCH(1,1) AIC: -1396.6342                   <===============================                                   
# GARCH(1,2) AIC: -1387.1524                                  
# GARCH(2,1) AIC: -1372.2451                                  
# GARCH(2,2) AIC: -1385.3612                                   
# GARCH(2,3) AIC: -1375.5014                                   
# GARCH(3,2) AIC: -1361.0431                                   
# GARCH(3,3) AIC: -1361.3381641705846                                   
# =============================================================================

# Nvidia
# =============================================================================
# GARCH(0,1) AIC: -848.0542                                   
# GARCH(1,0) AIC: -847.0045                                   
# GARCH(1,1) AIC: -849.6915                                   
# GARCH(1,2) AIC: -857.2097                   <===============================                                   
# GARCH(2,1) AIC: -848.0754                                   
# GARCH(2,2) AIC: -856.6277                                   
# GARCH(2,3) AIC: -855.2502444490707                                   
# GARCH(3,2) AIC: -855.2503                                   
# GARCH(3,3) AIC: -855.2640842634432                                   
# =============================================================================


#TestingSet
# Bitcoin
# =============================================================================
# GARCH(0,1) AIC: -497.6258                                  
# GARCH(1,0) AIC: -496.9131                                   
# GARCH(1,1) AIC: -500.3822                   <===============================                                 
# GARCH(1,2) AIC: -498.6274                                  
# GARCH(2,1) AIC: -494.3657                                   
# GARCH(2,2) AIC: -496.6074                                  
# GARCH(2,3) AIC: -490.0529                                  
# GARCH(3,2) AIC: -489.9158                                   
# GARCH(3,3) AIC: -488.4499                                   
# =============================================================================

# Gold
# =============================================================================
# GARCH(0,1) AIC: -644.4167                                   
# GARCH(1,0) AIC: -645.2839                   <===============================                                   
# GARCH(1,1) AIC: -645.2687                                  
# GARCH(1,2) AIC: -638.7674                                  
# GARCH(2,1) AIC: -638.3204                                    
# GARCH(2,2) AIC: -637.7608                                   
# GARCH(2,3) AIC: -629.7317                                   
# GARCH(3,2) AIC: -625.8882                                   
# GARCH(3,3) AIC: -629.6187                                   
# =============================================================================

# Nvidia
# =============================================================================
# GARCH(0,1) AIC: -428.2681                                   
# GARCH(1,0) AIC: -428.2682                   <===============================                                   
# GARCH(1,1) AIC: -426.2682                                   
# GARCH(1,2) AIC: -425.01191577892865                                   
# GARCH(2,1) AIC: -425.012                                   
# GARCH(2,2) AIC: -423.0108810714063                                   
# GARCH(2,3) AIC: -415.4993067241409                                   
# GARCH(3,2) AIC: -415.49931099983417                                   
# GARCH(3,3) AIC: -413.4993081912068                                   
# =============================================================================

