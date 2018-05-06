#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SQLContext
#sqlContext = SQLContext(sc)
from pyspark.sql import Row
import math
import pandas as pd
from pyspark.mllib.linalg import Vectors
import numpy as np
from pyspark.mllib.stat import Statistics
import matplotlib.pyplot as plt

"""tweetData = sc.textFile("clean_tweet.csv")
tweetData.collect()

parts=tweetData.map(lambda line: line.split(","))
textMap=parts.map(lambda p:Row(text=p[1]))
textMap.collect()
textDf=sqlContext.createDataFrame(textMap)
textPandas=textDf.toPandas()
polarityMap=parts.map(lambda p:Row(text=p[2]))
polarityMap.collect()
polarityDf=sqlContext.createDataFrame(polarityMap)
poralityPandas=polarityDf.toPandas()
"""
tweetData=pd.read_csv("clean_tweetfloat.csv")
#Split the dataset into TRain,Validation and Test Dataset

x = tweetData.text
y = tweetData.target
from sklearn.model_selection import train_test_split
SEED = 2000

x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)

x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, 
                 random_state=SEED)

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

trainDf=x_train.to_frame().join(y_train.to_frame())
validationDf=x_validation.to_frame().join(y_validation.to_frame())
testDf=x_test.to_frame().join(y_validation.to_frame())

trainDf.to_csv('train_dffloat.csv',encoding='utf-8')
validationDf.to_csv('validation_dffloat.csv',encoding='utf-8')
testDf.to_csv('test_dffloat.csv',encoding='utf-8')


from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time

#Function to Check the Accuracy of each ALgorithm

def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
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


cv = CountVectorizer()
lr = LogisticRegression()
n_features = np.arange(10000,100001,10000)

#N feature Acccuracy checker Function

def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
    result = []
    print (classifier)
    print ("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print ("Validation result for {} features".format(n))
        nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((n,nfeature_accuracy,tt_time))
    return result

csv = 'term_freq_df.csv'
term_freq_df = pd.read_csv(csv,index_col=0)
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

from sklearn.feature_extraction import text

a = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))
b = text.ENGLISH_STOP_WORDS
set(a).issubset(set(b))

my_stop_words = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))


print ("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
feature_result_wosw = nfeature_accuracy_checker(stop_words='english')
"""feature_result_wosw =[(10000, 0.75258896629636607, 131.19489359855652),
 (20000, 0.75509947906859975, 157.83297562599182),
 (30000, 0.75560158162304647, 205.67666673660278),
 (40000, 0.75673131237055169, 247.004465341568),
 (50000, 0.75648026109332833, 242.1042070388794),
 (60000, 0.75622920981610497, 242.92029118537903),
 (70000, 0.75654302391263417, 227.57554078102112),
 (80000, 0.75654302391263417, 227.56452989578247),
 (90000, 0.75717065210569257, 230.80325078964233),
 (100000, 0.75679407518985753, 265.546808719635)]
"""
print ("RESULT FOR UNIGRAM WITH STOP WORDS\n")
feature_result_ug = nfeature_accuracy_checker()
"""
feature_result_ug=[(10000, 0.77700370300633903, 221.2534363269806),
 (20000, 0.77807067093453841, 293.55897641181946),
 (30000, 0.77775685683800921, 315.86500883102417),
 (40000, 0.77832172221176177, 299.0194375514984),
 (50000, 0.77844724785037345, 374.40879487991333),
 (60000, 0.77813343375384425, 422.03608894348145),
 (70000, 0.77781961965731505, 398.57026743888855),
 (80000, 0.77788238247662089, 378.98844838142395),
 (90000, 0.77813343375384425, 481.2163510322571),
 (100000, 0.77775685683800921, 485.64167833328247)]

"""
print ("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words)
"""
feature_result_wocsw=[(10000, 0.7682796711228268, 156.0823574066162),
 (20000, 0.76834243394213264, 193.56152415275574),
 (30000, 0.77072742107575476, 230.35070705413818),
 (40000, 0.77053913261783724, 261.11463952064514),
 (50000, 0.77053913261783724, 308.4283673763275),
 (60000, 0.77097847235297812, 315.6927206516266),
 (70000, 0.77116676081089564, 317.7068431377411),
 (80000, 0.77141781208811899, 299.7355411052704),
 (90000, 0.77148057490742483, 352.36812925338745),
 (100000, 0.77185715182325987, 310.8561990261078)]
"""

nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='with stop words')
plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy,label='without custom stop words')
plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy,label='without stop words')
plt.title("Without stop words VS With stop words (Unigram): Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()

print ("RESULT FOR BIGRAM WITH STOP WORDS\n")
feature_result_bg=nfeature_accuracy_checker(ngram_range=(1, 2))
"""
feature_result_bg=[(10000, 0.78384485031067597, 481.7151062488556),
 (20000, 0.79043494633778955, 383.3442177772522),
 (30000, 0.79288269629071739, 462.0317659378052),
 (40000, 0.79426347831544597, 572.7171635627747),
 (50000, 0.79307098474863491, 565.5704348087311),
 (60000, 0.79376137576099914, 755.1762626171112),
 (70000, 0.79238059373627068, 680.3447363376617),
 (80000, 0.79181572836251801, 841.8471477031708),
 (90000, 0.79143915144668298, 945.9385240077972),
 (100000, 0.79281993347141155, 871.9716830253601)]
"""
nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy, label='with stop words')
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()

print ("RESULT FOR TRIGRAM WITH STOP WORDS\n")
feature_result_tg=nfeature_accuracy_checker(ngram_range=(1, 3))
"""
feature_result_tg=[(10000, 0.78164815163497148, 1762.043521642685),
 (20000, 0.78987008096403688, 2891.4169483184814),
 (30000, 0.79507939496642188, 3836.8182287216187),
 (40000, 0.79627188853323294, 3714.0135822296143),
 (50000, 0.79564426034017444, 5371.449221372604),
 (60000, 0.79300822192932907, 6610.117861032486),
 (70000, 0.79407518985752845, 5043.631604909897),
 (80000, 0.7925688821941882, 5863.750514745712),
 (90000, 0.792255068097659, 6526.677508354187),
 (100000, 0.79156467708529465, 6959.324209928513)]
"""
nfeatures_plot_tg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_bg.validation_accuracy, label='with stop words')
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()




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
    
tg_cv = CountVectorizer(max_features=80000,ngram_range=(1, 3))
tg_pipeline = Pipeline([
        ('vectorizer', tg_cv),
        ('classifier', lr)
    ])
train_test_and_evaluate(tg_pipeline, x_train, y_train, x_validation, y_validation)

#Comparison with TFIDF Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer()

feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec)
"""
feature_result_ugt=[(10000, 0.77625054917466896, 422.9538435935974),
 (20000, 0.77712922864495071, 143.37523102760315),
 (30000, 0.77832172221176177, 132.39003372192383),
 (40000, 0.77825895939245593, 160.7574586868286),
 (50000, 0.77844724785037345, 147.9197416305542),
 (60000, 0.77894935040482016, 171.19996619224548),
 (70000, 0.77888658758551432, 156.09799575805664),
 (80000, 0.77888658758551432, 179.3227412700653),
 (90000, 0.77857277348898513, 182.1217155456543),
 (100000, 0.77945145295926688, 184.35317373275757)]
"""
feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 2))
"""
feature_result_bgt=[(10000, 0.78114604908052465, 248.19150137901306),
 (20000, 0.78993284378334272, 232.10098838806152),
 (30000, 0.79438900395405765, 243.22714757919312),
 (40000, 0.79627188853323294, 250.5534689426422),
 (50000, 0.79702504236490301, 258.7881667613983),
 (60000, 0.79733885646143221, 276.6667687892914),
 (70000, 0.79602083725600958, 278.97376251220703),
 (80000, 0.79627188853323294, 277.5633125305176),
 (90000, 0.79815477311240823, 287.05480551719666),
 (100000, 0.79884516412477247, 289.1636571884155)]
"""
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,ngram_range=(1, 3))


#nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])
plt.figure(figsize=(8,6))
#plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='blue')
plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()


from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel

names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB", 
         "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
classifiers = [
    LogisticRegression(),
    LinearSVC(),
    Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LinearSVC(penalty="l2"))]),
    MultinomialNB(),
    BernoulliNB(),
    RidgeClassifier(),
    AdaBoostClassifier(),
    Perceptron(),
    PassiveAggressiveClassifier(),
    NearestCentroid()
    ]
zipped_clf = zip(names,classifiers)

tvec = TfidfVectorizer()
def classifier_comparator(vectorizer=tvec, n_features=10000, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
    result = []
    vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
    for n,c in classifier:
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', c)
        ])
        print ("Validation result for {}".format(n))
        print (c)
        clf_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((n,clf_accuracy,tt_time))

trigram_result = classifier_comparator(n_features=100000,ngram_range=(1,3))
"""
trigram_result=[('Logistic Regression', 0.79702504236490301, 1391.847366809845),
 ('Linear SVC', 0.79087428607293042, 2852.5113706588745),
 ('LinearSVC with L1-based feature selection',
  0.7913136258080713,
  3788.405289888382),
 ('Multinomial NB', 0.77612502353605728, 3464.317844390869),
 ('Bernoulli NB', 0.76250549174668925, 4009.3094704151154),
 ('Ridge Classifier', 0.79074876043431874, 3994.2744159698486),
 ('AdaBoost', 0.68041172409464634, 5310.314840555191),
 ('Perceptron', 0.72283938994539632, 8347.065608501434),
 ('Passive-Aggresive', 0.76294483148183012, 6820.882032632828),
 ('Nearest Centroid', 0.70940814661394591, 4047.1828265190125)]
"""

import pandas as pd

new_dataframe = pd.DataFrame(
      { "I.Algorithm Name" : ['Logistic Regression','Linear SVC','LinearSVC with L1-based feature selection',
                        'Multinomial NB','Bernoulli NB','Ridge Classifier','AdaBoost',
                        'Perceptron','Passive-Aggressive','Nearest Centroid'],
        "II.Accuracy" : [79.702504236490301,79.087428607293042,79.13136258080713,
                        77.612502353605728,76.250549174668925,79.074876043431874,
                        68.041172409464634,72.283938994539632,76.294483148183012,70.940814661394591],       
       "III.Train-Test Time(in secs)": [1391.847366809845,2852.5113706588745,3788.405289888382,3464.317844390869,
                         4009.3094704151154,3994.2744159698486,5310.314840555191,8347.065608501434,
                         6820.882032632828,4047.1828265190125]                  
      }
)
new_dataframe.to_csv('Tf-Idf_Algorithm_Accuracy')