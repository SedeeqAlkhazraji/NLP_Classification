"""
    @author: Sedeeq Al-khazraji, Will Thompson and Kruthika Simha
"""

from random import randint
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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
         "Decision Tree", "Random Forest", "Neural Net","AdaBoost","SGDClassifier","RidgeClassifier","VotingClassifier"]

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
               RidgeClassifier(tol=0.0001),
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
        if name == 'LogisticRegression':
            parameters = {'tol':[0.1,0.001,0.0001]}
            clf = GridSearchCV(clf, parameters)
        elif name == 'Nearest Neighbors':
            parameters = {'n_neighbors':[3,5,7,10]}
            clf = GridSearchCV(clf, parameters)
        elif name == 'Linear SVM':
            parameters = {'C':[0.001,0.1,1,10,100], 'tol':[0.1,0.01,0.001]}
            clf = GridSearchCV(clf, parameters)
        elif name == 'RBF SVM':
            parameters = {'C':[0.001,0.1,1,10,100], 'tol':[0.1,0.01,0.001]}
            clf = GridSearchCV(clf, parameters)
        elif name == 'Random Forest':
            parameters = {'n_classifiers':[1,10,100,250]}
            clf = GridSearchCV(clf, parameters)
        elif name == 'Neural Net':
            parameters = {'hidden_layer_sizes':range(200),'activation':('logistic','tanh','relu'), 'solvers':('lbfgs', 'sgd', 'adam'), 'learning_rate':('constant', 'invscaling','adaptive'),'learning_rate_init':[0.1,0.01,0.001]}
            clf = GridSearchCV(clf, parameters)
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
