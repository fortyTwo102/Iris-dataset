import os
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

data = np.loadtxt('dataset.txt', delimiter=',')
X, y = data[:, 0:4], data[:, 4]

print("Shape of X", X.shape)
print("Shape of y", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

svm = SVC(verbose=True,kernel='linear',gamma=0.1,C=1)
parameters = {'kernel':('linear', 'rbf','poly'), 'C':(1,2,0.5,0.75),'gamma': (0.1,0.001,0.01,1,2,'auto')}
clf = GridSearchCV(svm, parameters,cv=5)
clf.fit(X_train,y_train)

print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)

y_pred = clf.predict(X_test)

y_pred = y_pred.reshape(y_pred.shape[0],-1)
y_test = y_test.reshape(y_test.shape[0],-1)

print("Shape of y_pred ", y_pred.shape)
print("Shape of y_test ", y_test.shape)

print("Accuracy: ", float(clf.score(X_test, y_test)))
print("f1_score: ", f1_score(y_pred,y_test,average='macro'))
