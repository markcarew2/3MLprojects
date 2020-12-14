import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

open("output2.csv", "w").close()
        
pd.options.display.precision = 16

#Import Data into DataFrame and intialize alpha
df = pd.read_csv("input2.csv", header=None)
allAs = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)

#Carve out the label so we don't scale it
labels = df.iloc[:,-1]
lastColumn = len(df.columns) - 1
df.drop(lastColumn, axis =1, inplace = True)
labelMean = labels.sum() / len(labels)

#Find Means and stdevs of columns
meanSeries = df.mean(axis=0) * -1
array = df.to_numpy()
sdArray = np.std(array, axis = 0)
sdSeries = pd.Series(sdArray)

#Normalize and add intercept column B0
df1 = df.add(meanSeries, axis = 1)
dfNormalized = df1.divide(sdSeries, axis = 1)
dfNormalized[len(dfNormalized.columns)] = 1

#Initialize alpha and Betas
for a in allAs:
    Bs = pd.Series({0:0,1:0,2:0})

    #Calculate Change for Each B
    for n in range(100):
        examplesTimesBs = dfNormalized.multiply(Bs, axis = 1)
        fxs = examplesTimesBs.sum(axis=1)
        fxsminusys = fxs.subtract(labels)
        vectorMatrix = dfNormalized.multiply(fxsminusys, axis=0)
        changeSeries = vectorMatrix.sum(axis =0) * a
        changeSeries = changeSeries / len(dfNormalized.index)

        Bs = Bs.subtract(changeSeries)
    
    RSS = (fxsminusys ** 2).sum()
    
    output = str(a) + ",100," + str(Bs[2]) + "," + str(Bs[0]) + "," + str(Bs[1]) + "\n"

    with open("output2.csv", "a") as f:
        f.write(output)

    xList = np.arange(-2, 2,.666 * .1)
    xList = xList[0:60]
    yList = np.arange(-2,4,.1)
    X, Y = np.meshgrid(xList,yList)

    Z = X * Bs[0] + Y * Bs[1] + Bs[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.plot(dfNormalized[0],dfNormalized[1], labels,  'ro')
    ax.plot_surface(X,Y,Z)
    learningRate = str(a)
    title = "Learning Rate: " + learningRate
    plt.title(title)
    plt.show(block=True)



iters = 100
for n in range(iters):
    a = 1
    Bs = pd.Series({0:0,1:0,2:0})
    examplesTimesBs = dfNormalized.multiply(Bs, axis = 1)
    fxs = examplesTimesBs.sum(axis=1)
    fxsminusys = fxs.subtract(labels)
    vectorMatrix = dfNormalized.multiply(fxsminusys, axis=0)
    changeSeries = vectorMatrix.sum(axis =0) * a
    changeSeries = changeSeries / len(dfNormalized.index)

    Bs = Bs.subtract(changeSeries)

output = str(a) + "," + str(iters) + "," + str(Bs[2]) + "," + str(Bs[0]) + "," + str(Bs[1])
with open("output2.csv", "a") as f:
    f.write(output)