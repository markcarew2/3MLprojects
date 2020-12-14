import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

#Helper function to match columns of an example to ws and add
def linearFunctionHelper(series1, series2, y):
    global changed
    global ws
    
    xs = pd.Series([series1[0], series1[1],1])

    series3 = xs.mul(series2)

    value = series3.sum()
    v = value * y
    
    if v <= 0:
        changeVector = series1 * y
        changeVector.iloc[-1] = y
        #print("Change Vector is: ", changeVector)
        #print("Ws are: ",series2)
        ws = series2.add(changeVector)
        changed += 1
    else:
        pass

open("output1.csv", "w").close()
        

#Import Data into DataFrame
df = pd.read_csv("input1.csv", header=None)

#Initialize ws to 0
ws = pd.Series({2:0, 0:0, 1:0})

#Loop until convergence:
while(True):
    changed = 0
    #Go through all examples, change ws if f(x) != y
    df.apply(lambda x: linearFunctionHelper(x, ws, x.iloc[-1]), axis=1)

    #Check convergence
    if changed==0:
        break

    #Append ws to output file
    with open("output1.csv", "a") as f:
        listy = ws.values.tolist()
        listy = list(map(str, listy))
        f.write(",".join(listy))
        f.write("\n")
    
        df1 = df[df.iloc[:,-1] > 0]
    df2 = df[df.iloc[:,-1] < 0]

    w1List = np.arange(0.0, 16.0, .5)

    w2List = w1List * ws[0]
    w2List = w2List + ws[2]
    w2List = w2List/ws[1]
    w2List = w2List * -1

    plt.plot(df1[0],df1[1], 'ro')
    plt.plot(df2[0],df2[1],'bo')
    plt.plot(w1List, w2List)

    plt.show(block=False)
    plt.pause(.5)
    plt.close()



df1 = df[df.iloc[:,-1] > 0]
df2 = df[df.iloc[:,-1] < 0]

w1List = np.arange(0.0, 16.0, .5)

w2List = w1List * ws[0]
w2List = w2List + ws[2]
w2List = w2List/ws[1]
w2List = w2List * -1

plt.plot(df1[0],df1[1], 'ro')
plt.plot(df2[0],df2[1],'bo')
plt.plot(w1List, w2List)

plt.show()