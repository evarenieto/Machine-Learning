#!/usr/bin/python

"""
    
    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
#########################################################
clf = DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf = clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
acc = accuracy_score(pred, labels_test)
print acc
print "Decision Tree accuracy: %r" % acc
print "no. of features in your data: %r" % len(features_train[0])
###################################################################

################################################
############## results ########################
################ here ##########################
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#training time: 59.121 s
#prediction time: 0.052 s
#0.977246871445
#Decision Tree accuracy: 0.9772468714448237
#no. of features in your data: 3785
##################################################
############### and also #########################
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#training time: 5.211 s
#prediction time: 0.002 s
#0.967007963595
#Decision Tree accuracy: 0.9670079635949943
#no. of features in your data: 379
##################################################
