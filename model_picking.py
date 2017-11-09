"""
    @author: Sedeeq Al-khazraji
    @co-author: Will Thompson & Kruthika Simha 
"""

from random import randint
import numpy as np
import pandas as pd
import nltk, string, time, csv, string
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pprint import pprint
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle


#Prepare variables
DataSet = "PS3_training_data.txt"
DataSetHeaders = ['ID', 'TEXT', 'SENTIMENT', 'CATEGORY', 'GENRE']
TASKS = ['GENRE', 'CATEGORY', 'SENTIMENT']
clf_training_df = pd.DataFrame(columns=['clf', 'GENRE', 'SENTIMENT', 'CATEGORY'])
clf_testing_df = pd.DataFrame(columns=['clf', 'GENRE', 'SENTIMENT', 'CATEGORY'])

bestModel ={'GENRE':0, 'CATEGORY':0,'SENTIMENT':0}

# Prepare Classifier
# Here we need to work on different options
# Check this link for mor detiles: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

names = ["LogisticRegression","Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net","AdaBoost","SGDClassifier","VotingClassifier"]

classifierslst = [
    ("LogisticRegression" , LogisticRegression(random_state=1)),
    ("SVC2", SVC(gamma=2, C=1)),
    ("NN", MLPClassifier(alpha=1,hidden_layer_sizes=(5,10,5))),
    ("SGDClassifier", SGDClassifier()),
    ("DecisionTreeClassifier", DecisionTreeClassifier())]

classifiers = [
    LogisticRegression(random_state=1),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    SGDClassifier(),
    VotingClassifier(estimators=classifierslst, voting='hard', n_jobs=-1)
]

#Load DataSet
df = pd.read_csv(DataSet, sep='\t', index_col=False, header=None, names=DataSetHeaders)
print ("Number of data points: %d" % len(df))
print ("Random output:", df.iloc[1]['TEXT'])

#Split DataSet
train_data, test_data = model_selection.train_test_split(df, test_size = 0.2)
print ("train_data len = " , len(train_data))
print ("test_data len = " , len(test_data))
#Only consider the sentences which are not NONE in topic
Task3_train_data, Task3_test_data = model_selection.train_test_split(df.loc[df['GENRE'] == 'GENRE_A'], test_size = 0.2)
print ("Task3_train_data len = " , len(Task3_train_data))
print ("Task3_test_data len = " , len(Task3_test_data))

#TfidfVectorizer  combines all the options of CountVectorizer and TfidfTransformer in a single model
#http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.70, stop_words='english',max_features=1000)

#MajorityVotingClassifier = VotingClassifier(estimators=classifiers[:], voting='hard', n_jobs=-1)
#classifiers.append(MajorityVotingClassifier)

for task in TASKS:
    Y_train = train_data[task]
    Y_test = test_data[task]
    X_train = vectorizer.fit_transform(train_data['TEXT'])
    X_test = vectorizer.transform(test_data['TEXT'])

    if task == 'CATEGORY':
        Y_train = Task3_train_data[task]
        Y_test = Task3_test_data[task]
        X_train = vectorizer.fit_transform(Task3_train_data['TEXT'])
        X_test = vectorizer.transform(Task3_test_data['TEXT'])


    # iterate over classifiers
    bestModel_acc = 0
    for name, clf in zip(names, classifiers):
        if task == "SENTIMENT":
            clf.class_weight = {"NEGATIVE":1,"POSITIVE":1,"NEUTRAL":5}
        kfold = model_selection.KFold(n_splits=10, random_state=randint(1, 10))
        clf.fit(X_train, Y_train)

        score = cross_val_score(clf,X_train, Y_train, cv=kfold)
        acc = score.mean()
        print("Accuracy score for %s in %s task is %f" % (name, task, acc))
        clf_training_df=clf_training_df.append({"clf": name, task:  acc}, ignore_index=True)
        # "Generalization Error"
        score_test = clf.score(X_test, Y_test)
        acc_test = score_test.mean()
        clf_testing_df=clf_testing_df.append({"clf": name, task:  acc_test}, ignore_index=True)

        if acc > bestModel_acc:
            bestModel_acc=acc
            bestModel_test_acc = acc_test
            best_clf = clf
            best_clf_name=name


    bestModel[task] = best_clf
    print ("Best classifier for task %s is %s, The accuraccy of the model is %f (Trainging accuracy is %f)" %(task,best_clf_name,bestModel_test_acc, bestModel_acc))
    # save the model to disk
    #filename = task + '_' + best_clf_name + '_finalized_model.sav'
    filename = task + '_finalized_model.sav'
    pickle.dump(best_clf, open(filename, 'wb'))


print ("Training Accuracy:")
clf_training_df = clf_training_df.groupby(['clf']).first().reset_index()
print (clf_training_df)

print ("Testing Accuracy:")
clf_testing_df = clf_testing_df.groupby(['clf']).first().reset_index()
print (clf_testing_df )




'''
# load the model from disk
filename = task + '_finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
'''
