import numpy as np
from sklearn.dummy import DummyClassifier
import copy

class ClassifierChain:
    def __init__(self, baseline_clf):
        self.clf = baseline_clf
    
    def fit(self, X, Y):
        self.perm = np.random.permutation(len(Y[0]))
        Y = Y[:,self.perm]
        self.classifiers = []
        for p in range(len(self.perm)):
            X_p = X[~np.isnan(Y[:,p])]
            Y_p = Y[~np.isnan(Y[:,p]), p]
            if len(X_p) == 0: # all labels are nan -> predict 0
                clf = DummyClassifier(strategy='constant', constant=0)
                self.classifiers.append(clf.fit([X[0]], [0]))
            elif len(np.unique(Y_p)) == 1: # only one class -> predict that class
                clf = DummyClassifier(strategy='most_frequent')
                self.classifiers.append(clf.fit(X_p, Y_p))
            else:
                clf = copy.deepcopy(self.clf)
                self.classifiers.append(clf.fit(X_p, Y_p))
            newcol = Y[:,p]
            pred = clf.predict(X)
            newcol[np.isnan(newcol)] = pred[np.isnan(newcol)] # fill in missing values with clf predictions
            X = np.column_stack((X, newcol))
            
    def predict(self, X):
        labels = np.empty((len(X), 0))
        for clf in self.classifiers:
            pred = clf.predict(np.column_stack((X, labels)))
            labels = np.column_stack((labels, pred))
        return labels[:, np.argsort(self.perm)]
    
    def predict_proba(self, X):
        labels = np.empty((len(X), 0))
        for clf in self.classifiers:
            pred = clf.predict_proba(np.column_stack((X, np.round(labels))))
            if pred.shape[1] > 1: pred = pred[:,1]
            else: pred = pred * clf.predict(np.column_stack(([X[0]], np.round([labels[0]]))))[0]
            labels = np.column_stack((labels, pred))
        return labels[:, np.argsort(self.perm)]
    
class EnsembleClassifierChain:
    def __init__(self, baseline_clf, num_classifiers=10):
        self.clf = baseline_clf
        self.num = num_classifiers
    
    def fit(self, X, Y):
        self.classifiers = []
        self.num_labels = len(Y[0])
        for p in range(self.num):
            clf = ClassifierChain(self.clf)
            clf.fit(X, Y)
            self.classifiers.append(clf)
            
    def predict(self, X):
        labels = np.zeros((len(X), self.num_labels))
        for clf in self.classifiers:
            labels += clf.predict(X)
        return np.round(labels / self.num)
    
    def predict_proba(self, X):
        labels = np.zeros((len(X), self.num_labels))
        for clf in self.classifiers:
            labels += clf.predict_proba(X)
        return labels / self.num
    
def hamming_accuracy(Y, Y_hat):
    non_nans = np.count_nonzero(~np.isnan(Y))
    return 1 if non_nans==0 else np.sum(Y==Y_hat) / non_nans

def confusion_matrix(Y, Y_hat):
    TP = np.sum((Y==1)&(Y_hat==1))
    FP = np.sum((Y==0)&(Y_hat==1))
    TN = np.sum((Y==0)&(Y_hat==0))
    FN = np.sum((Y==1)&(Y_hat==0))
    
    return np.array([[TP, FN], [FP, TN]])