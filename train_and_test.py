import numpy as np
import scipy.io as spio
import sys
from os import path
from sklearn import svm
from sklearn import grid_search
from sklearn.externals import joblib
from time import localtime, strftime


def one_vs_all_svm(labels, Xtrain, expression, kfold=5):
    # expression 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    # kfold cross validation
    y = np.array( labels == expression, dtype=np.int) # convert from multiple class label to binary
    # parameter grids
    parameters = { 'C': [1, 2, 10, 1e2, 1e3, 1e6], 'gamma': [1e-3, 1e-2, 0.1, 0.5]}  # parameter costs of misclassification and gaussian kernel gammas
    # C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. smaller C prefer larger hyper-plane margin
    # smaller gamma generate lower bias but higher variance

    # Do a cross validation grid search, using ROC Area under Curve as the score
    classifier = grid_search.GridSearchCV(svm.SVC(kernel='rbf',  probability=True), parameters, cv=kfold, scoring="roc_auc", verbose=2) # Turn on probability prediction
    classifier.fit(Xtrain, y)

    # return the classifier who maintains the best estimators parameters
    return classifier


def train_svm_predictors(Ytrain, Xtrain, pathprefix):
    classifiers = []
    for e in range(7): # left neutral out. 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        print 'Training expression ' + str(e) + ' @ ' + strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())
        clf = one_vs_all_svm(Ytrain, Xtrain, e)
        joblib.dump(clf, pathprefix+str(e)+'.clf') # Persist model whenever possible. Since it is very computationally expensive.
        classifiers.append(clf)

    return classifiers


def model_apply(classifiers, Ytest, Xtest):
    prob = np.array([clf.predict_proba(Xtest) for clf in classifiers])
    prob = prob[:, :, 1]  # get the probability of "True"
    predict =  np.argmax(prob, axis=0)  # get the index of max prob, which is also the expression id

    return predict, predict==Ytest # return predicted labels and error



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print 'Usage: python training.py path/to/working/folder'
        exit(1)

    # load labels (Y) and samples with selected features (X)
    Ytrain =  spio.loadmat(path.join(sys.argv[2], 'train_labels.mat'))['labels'][0]  # need this index '0' go the 1-D array
    Ytest =  spio.loadmat(path.join(sys.argv[2], 'test_labels.mat'))['labels'][0]  # need this index '0' go the 1-D array
    Xtrain = spio.loadmat(path.join(sys.argv[1], "xtrain.mat")['X'])
    Xtest = spio.loadmat(path.join(sys.argv[1], "xtest.mat")['X'])

    # randomly shuffle the training data
    indices = np.random.permutation(Ytrain.shape[0])
    Ytrain = Ytrain[indices]
    Xtrain = Xtrain[indices]

    classifieres = train_svm_predictors(Ytrain, Xtrain, path.join(sys.argv[1], 'expression_'))
    predict, predict_error = model_apply(classifieres, Ytest, Xtest)
    print "Error rate = " + str(sum(predict_error)/len(predict))