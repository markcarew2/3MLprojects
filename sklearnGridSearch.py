import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import svm, neighbors, linear_model, tree, ensemble
from itertools import product

df = pd.read_csv("input3.csv")

trainDF, testDF = train_test_split(df, train_size = .6, stratify = df["label"], random_state = 1)

#Grid search automatically uses kFOld validation but below is dividing manually, for reference
#Just iterate through the split data object to extract the various divisions
#kf = KFold()
#splitData = kf.split(trainDF)

#Make the classifiers of different types
svr = svm.SVC()
knn = neighbors.KNeighborsClassifier()
lgrg = linear_model.LogisticRegression()
dTree = tree.DecisionTreeClassifier()
rForest = ensemble.RandomForestClassifier()

#Set up the various gridsearches we're going to do
linearClassifier = GridSearchCV(svr, {'kernel': ['linear'], 'C':[0.1, 0.5, 1, 5, 10, 50, 100]})
polyClassifier = GridSearchCV(svr, {'kernel': ['poly'], 'C':[0.1, 1, 3], 'degree':[4, 5, 6],'gamma':[0.1,0.5]})
rbfClassifier = GridSearchCV(svr, {'kernel':['rbf'], 'C':[0.1, 0.5, 1, 5, 10, 50, 100], 'gamma':[0.1, 0.5, 1, 3, 6, 10]})
knnClassifier = GridSearchCV(knn, {'algorithm': ['kd_tree'],'n_neighbors': [a+1 for a in range(50)], 'leaf_size':[a for a in range(5,65,5)]})
lgrgClassifier = GridSearchCV(lgrg, {'C': [0.1, 0.5, 1, 5, 10, 50, 100]})
rFClassifier = GridSearchCV(rForest, {'max_depth': [a+1 for a in range(50)], 'min_samples_split': [a for a in range(2,11)]})
dTreeClassifier = GridSearchCV(dTree, {'max_depth': [a+1 for a in range(50)], 'min_samples_split': [a for a in range(2,11)]})

#Set up variables for graphing
#We'll produce graphs after every classifier 
#and a summary graph at the end
x_min, x_max = testDF.iloc[:,0].min() - 1, testDF.iloc[:,0].max() + 1
y_min, y_max = testDF.iloc[:,1].min() - 1, testDF.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


#Train the Linear Classifier
linearClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
#print(linearClassifier.cv_results_)
print("Best Parameters: ", linearClassifier.best_estimator_)
bestLinScore = linearClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print("SVM Linear Classifier Train Score: ", bestLinScore)
linTest = linearClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print("Test Score: ", linTest)

#Graph the Linear Classifier with Test Data
Z = linearClassifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.4)
plt.scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
title = linearClassifier.best_estimator_
plt.title(title)
scores = "Test Score: " + str(linTest)
plt.xlabel(scores)
plt.show()


#Repeat With other classifiers

#Train Poly
polyClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
#print(polyClassifier.cv_results_)
print("\nBest Parameters: ", polyClassifier.best_estimator_)
bestPolyScore = polyClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print("Polynomial SVM Train Score: ", bestPolyScore)
polyScore = polyClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print("Test Score: ", polyScore)

#Graph Poly
Z = polyClassifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.4)
plt.scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
title = polyClassifier.best_estimator_
plt.title(title)
scores = "Test Score: " + str(polyScore)
plt.xlabel(scores)
plt.show()

#Train RBF
rbfClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
#print(rbfClassifier.cv_results_)
print("\nBest Parameters: ", rbfClassifier.best_estimator_)
bestrbfScore = rbfClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print("RBF SVM Train Score: ", bestrbfScore)
rbfTest = rbfClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print("Test Score: ", rbfTest)

#Graph RBF
Z = rbfClassifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.4)
plt.scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
title = rbfClassifier.best_estimator_
plt.title(title)
scores = "Test Score: " + str(rbfTest)
plt.xlabel(scores)
plt.show()

#Train KNN Classifier
knnClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
#print(knnClassifier.cv_results_)
print("\nBest Parameters: ", knnClassifier.best_estimator_)
bestknnScore = knnClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print("K-Nearest Neighbours Train Score: ",bestknnScore)
knnTest = knnClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print("Test Score: ", knnTest)

#Graph KNN Classifier
Z = knnClassifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.4)
plt.scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
title = knnClassifier.best_estimator_
plt.title(title)
scores = "Test Score: " + str(knnTest)
plt.xlabel(scores)
plt.show()

#Train Logistic Regression Classifier
lgrgClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
#print(lgrgClassifier.cv_results_)
print("\nBest Parameters: ", lgrgClassifier.best_estimator_)
bestlgrgScore = lgrgClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print("Logistic Regression Train Score: ", bestlgrgScore)
lgrgTest = lgrgClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print("Test Score: ", lgrgTest)

#Graph Logistic regression
Z = lgrgClassifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.4)
plt.scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
title = lgrgClassifier.best_estimator_
plt.title(title)
scores = "Test Score: " + str(lgrgTest)
plt.xlabel(scores)
plt.show()

#Train Random Forest Classifier
rFClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
#print(rFClassifier.cv_results_)
print("\nBest Parameters: ", rFClassifier.best_estimator_)
bestrFScore = rFClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print("Random Forest Train Score: ",bestrFScore)
rFTest = rFClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print("Test Score: ", rFTest)

#Graph Random Forest
Z = rFClassifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.4)
plt.scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
title = rFClassifier.best_estimator_
plt.title(title)
scores = "Test Score: " + str(rFTest)
plt.xlabel(scores)
plt.show()

#Train Decision Tree Classifier
dTreeClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
#print(dTreeClassifier.cv_results_)
print("\nBest Parameters: ", dTreeClassifier.best_estimator_)
bestdTreeScore = dTreeClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print("Decision Tree Train Score: ",bestdTreeScore)
dTreeTest = dTreeClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print("Test Score: ", dTreeTest)

#Graph Decision Tree Classifier
Z = dTreeClassifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.4)
plt.scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
title = dTreeClassifier.best_estimator_
plt.title(title)
scores = "Test Score: " + str(dTreeTest)
plt.xlabel(scores)
plt.show()


#Set up the summary graph
fig, ax = plt.subplots(4,2, sharex="col", sharey='row', figsize = (18,12))
plt.subplots_adjust(hspace=.4)

prod = product([0,1,2,3],[0,1])

classifiers = [dTreeClassifier, rFClassifier,lgrgClassifier,knnClassifier,rbfClassifier,polyClassifier,linearClassifier]
titles = ["Decision Tree", "Random Forest", "Logistic R", "KNN", "RBF", "Polynomial", "Linear"]
testScores = [dTreeTest, rFTest, lgrgTest, knnTest, rbfTest, polyScore, linTest]

#Populate the Subplots
for idx, clf, tt, score in zip(prod, classifiers, titles, testScores):

    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax[idx[0],idx[1]].contourf(xx, yy, Z, alpha=.4)
    ax[idx[0],idx[1]].scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
    ax[idx[0],idx[1]].set_title(tt)
    scores = "Test Score: " + str(score)
    ax[idx[0],idx[1]].set_xlabel(scores)

plt.show()