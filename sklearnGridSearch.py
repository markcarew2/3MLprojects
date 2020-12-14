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

svr = svm.SVC()
knn = neighbors.KNeighborsClassifier()
lgrg = linear_model.LogisticRegression()
dTree = tree.DecisionTreeClassifier()
rForest = ensemble.RandomForestClassifier()


linearClassifier = GridSearchCV(svr, {'kernel': ['linear'], 'C':[0.1, 0.5, 1, 5, 10, 50, 100]})
polyClassifier = GridSearchCV(svr, {'kernel': ['poly'], 'C':[0.1, 1, 3], 'degree':[4, 5, 6],'gamma':[0.1,0.5]})
rbfClassifier = GridSearchCV(svr, {'kernel':['rbf'], 'C':[0.1, 0.5, 1, 5, 10, 50, 100], 'gamma':[0.1, 0.5, 1, 3, 6, 10]})
knnClassifier = GridSearchCV(knn, {'algorithm': ['kd_tree'],'n_neighbors': [a+1 for a in range(50)], 'leaf_size':[a for a in range(5,65,5)]})
lgrgClassifier = GridSearchCV(lgrg, {'C': [0.1, 0.5, 1, 5, 10, 50, 100]})
rFClassifier = GridSearchCV(rForest, {'max_depth': [a+1 for a in range(50)], 'min_samples_split': [a for a in range(2,11)]})
dTreeClassifier = GridSearchCV(dTree, {'max_depth': [a+1 for a in range(50)], 'min_samples_split': [a for a in range(2,11)]})

linearClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(linearClassifier.cv_results_)
print(linearClassifier.best_estimator_)
bestLinScore = linearClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(bestLinScore)
linTest = linearClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])


polyClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(polyClassifier.cv_results_)
print(polyClassifier.best_estimator_)
bestPolyScore = polyClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(bestPolyScore)
polyScore = polyClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])

rbfClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(rbfClassifier.cv_results_)
print(rbfClassifier.best_estimator_)
bestrbfScore = rbfClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(bestrbfScore)
rbfTest = rbfClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print(rbfTest)

knnClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(knnClassifier.cv_results_)
print(knnClassifier.best_estimator_)
bestknnScore = knnClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(bestknnScore)
knnTest = knnClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])
print(knnTest)

lgrgClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(lgrgClassifier.cv_results_)
print(lgrgClassifier.best_estimator_)
bestlgrgScore = lgrgClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(bestlgrgScore)
lgrgTest = lgrgClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])

rFClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(rFClassifier.cv_results_)
print(rFClassifier.best_estimator_)
bestrFScore = rFClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(bestrFScore)
rFTest = rFClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])

dTreeClassifier.fit(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(dTreeClassifier.cv_results_)
print(dTreeClassifier.best_estimator_)
bestdTreeScore = dTreeClassifier.score(trainDF.iloc[:,[0,1]], trainDF.iloc[:,2])
print(bestdTreeScore)
dTreeTest = dTreeClassifier.score(testDF.iloc[:,[0,1]], testDF.iloc[:,2])

x_min, x_max = testDF.iloc[:,0].min() - 1, testDF.iloc[:,0].max() + 1
y_min, y_max = testDF.iloc[:,1].min() - 1, testDF.iloc[:,1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

fig, ax = plt.subplots(4,2, sharex="col", sharey='row', figsize = (14,8))

prod = product([0,1,2,3],[0,1])

classifiers = [dTreeClassifier, rFClassifier,lgrgClassifier,knnClassifier,rbfClassifier,polyClassifier,linearClassifier]
titles = ["Decision Tree", "Random Forest", "Logistic R", "KNN", "RBF", "Polynomial", "Linear"]

for idx, clf, tt in zip(prod, classifiers, titles):

    print(idx)
    print(clf)
    print(tt)
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax[idx[0],idx[1]].contourf(xx, yy, Z, alpha=.4)
    ax[idx[0],idx[1]].scatter(testDF.iloc[:,0], testDF.iloc[:,1],c=testDF.iloc[:,2])
    ax[idx[0],idx[1]].set_title(tt)

plt.show()