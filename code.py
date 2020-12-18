"""

for trainIndex, validIndex in splitData:
    trainData = trainDF.iloc[trainIndex, :]
    validData = trainDF.iloc[validIndex, :]

    linearClassifier.fit(trainData.iloc[:,[0,1]], trainData.iloc[:,2])
    linscore = linearClassifier.score(validData.iloc[:,[0,1]], validData.iloc[:,2])
    print(linearClassifier.best_params_)

    #polyClassifier.fit(trainData.iloc[:,[0,1]], trainData.iloc[:,2])
    #polyscore = polyClassifier.score(validData.iloc[:,[0,1]], validData.iloc[:,2])
    #print(polyClassifier.best_params_)

    rbfClassifier.fit(trainData.iloc[:,[0,1]], trainData.iloc[:,2])
    rbfscore = rbfClassifier.score(validData.iloc[:,[0,1]], validData.iloc[:,2])
    print(rbfClassifier.best_params_)

"""