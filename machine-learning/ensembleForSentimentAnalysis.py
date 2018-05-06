#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:32:59 2018

@author: arjun
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
import pandas as pd
import numpy as np


tweetData=pd.read_csv("clean_tweetfloat.csv")

x = tweetData.text
y = tweetData.target
from sklearn.model_selection import train_test_split
from textblob import TextBlob

SEED = 2000

x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)

x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, 
                 random_state=SEED)
header = ['id','text']
x_test.to_csv('testData.csv' , header=True)

print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive"
                           .format(len(x_train),   
                          (len(x_train[y_train == 0.0]) / (len(x_train)*1.))*100,  
                          (len(x_train[y_train == 1.0]) / (len(x_train)*1.))*100))

print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive"
       .format(len(x_validation),  
                        (len(x_validation[y_validation == 0.0]) / (len(x_validation)*1.))*100,   
                        (len(x_validation[y_validation == 1.0]) / (len(x_validation)*1.))*100))

print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive"
                       .format(len(x_test),   
                        (len(x_test[y_test == 0.0]) / (len(x_test)*1.))*100,     
                        (len(x_test[y_test == 1.0]) / (len(x_test)*1.))*100))

tbresult = [TextBlob(i).sentiment.polarity for i in x_validation]
tbpred = [0 if n < 0 else 1 for n in tbresult]

conmat = np.array(confusion_matrix(y_validation, tbpred, labels=[1,0]))

confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                         columns=['predicted_positive','predicted_negative'])

print ("Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbpred)*100))
print( "-"*80)
print ("Confusion Matrix\n")
print (confusion)
print( "-"*80)
print ("Classification Report\n")
print (classification_report(y_validation, tbpred))

def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    joblib.dump(sentiment_fit,'classifier_myModel.pkl')
    #with open('tweetsSentiModel', 'wb') as f:
        #pickle.dump(sentiment_fit, f)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    #filename = 'finalized_model.sav'
    #pickle.dump(sentiment_fit, open(filename, 'wb'))
    accuracy = accuracy_score(y_test, y_pred)
    print ("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print ("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print ("model has the same accuracy with the null accuracy")
    else:
        print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print ("train and test time: {0:.2f}s".format(train_test_time))
    print ("-"*80)
    return accuracy, train_test_time

def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))
    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                         columns=['predicted_negative','predicted_positive'])
    print ("null accuracy: {0:.2f}%".format(null_accuracy*100))
    print ("accuracy score: {0:.2f}%".format(accuracy*100))
    if accuracy > null_accuracy:
        print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
    elif accuracy == null_accuracy:
        print ("model has the same accuracy with the null accuracy")
    else:
        print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
    print ("-"*80)
    print ("Confusion Matrix\n")
    print (confusion)
    print ("-"*80)
    print ("Classification Report\n")
    print (classification_report(y_test, y_pred, target_names=['negative','positive']))
    
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier


from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer()
from time import time
import pickle

clf1 = LogisticRegression()
clf2 = LinearSVC()
clf3 = MultinomialNB()
clf4 = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5, 2),random_state=1)
clf5 = SGDClassifier(loss="hinge", penalty="l2")

eclf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('mnb', clf3), ('mlp', clf4), ('sgd',clf5)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Logistic Regression', 'Linear SVC', 'Multinomial NB', 'MLP Classifier', 'SGD Classifier', 'Ensemble']):
    checker_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=100000,ngram_range=(1, 3))),
            ('classifier', clf)
        ])
    print ("Validation result for {}".format(label))
    print (clf)
clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)


#MLP
clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5, 2),random_state=1)
checker_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=100000,ngram_range=(1, 3))),
            ('classifier', clf)
        ])
label='MLP Classifier'
print ("Validation result for {}".format(label))
print (clf)
clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)

#SGD
clf = SGDClassifier(loss="hinge", penalty="l2")
checker_pipeline = Pipeline([
          ('vectorizer', TfidfVectorizer(max_features=100000,ngram_range=(1, 3))),
          ('classifier', clf)
            ])
label='SGD Classifier'
print ("Validation result for {}".format(label))
print (clf)
clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
    


