# NLP Classification
> All classification tasks can be divided into two parts: Feature selection and extraction, and then the classification. The feature selection and extraction step include both cleaning up the data before classification, as well as picking out the best features from a pool of possible features. The classification step includes using a classifier to classify the data into a set of classes, and then fine- tuning the classification based on provided ground truth data.

## 1.   Libraries Used:
The scikit-learn library was used to implement the feature extraction as well as the classifiers. The tf-idf features were used for classification along with the following classifiers:
* a.   Logistic regression classifier b.   K-nearest neighbor classifier c.   Linear SVM classifier
* d.   RBF SVM classifier
* e.   Decision tree classifier
* f.    Random forests classifier
* g.   Multi-layer perceptron classifier h.   AdaBoost classifier
* i.    Stochastic gradient descent classifier
* j.    Ridge classifier
    * k.   A voting classifier with the following classifiers contributing to the vote, with all the classifiers having equal importance:
    * a.   Logistic regression classifier b.   RBF SVM classifier
    * c.   Stochastic gradient descent classifier d.   Random forests classifier
    * e.   Ridge classifier
    * f.    Decision tree classifier


## 2.   Feature Extraction:

All the data points with the label 'NONE' was removed from the dataset. The TfidfVectorizer function converts a collection of raw documents into a matrix with the tf-idf features. The top 1000 tf-idf features were used for classification.

## 3.   Train and Development set data Split:

The dataset was divided into train and development set with a 80:20 ratio using the model_selection.train_test_split function in scikit-learn.

## 4.   Steps for running the program:
The files that are provided are:
* a)  model_picking.py - The training code
* b)  test_classifiers.py - The testing code


There are two ways to run the model:
o    One way is to run model_picking.py and then run test_classifiers.py - this trains on the training data and tests the testing data on the recently trained model.
o    The other is to just run test_classifiers.py - this will load the provided final models for the three categories and test the data on these models.

When test_classifiers.py begins, a pop-up window will appear, where you can select the test dataset. The code then tests the provided data on the three models, and outputs the final accuracy as well as the confusion matrix for the three classifiers. A second pop-up window will prompt you to select the destination as well as provide a filename, for the storing the final predictions of the three models.

## Contributing
* **Bushra Aloraini** {baloraini@uwaterloo.ca} and  **Meiyappan Nagappan** {mei.nagappan@uwaterloo.ca}

## Authors

* **Sedeeq Al-khazraji** - *Initial work* -
sha6709@rit.edu
https://github.com/SedeeqAlkhazraji

## License
This project is licensed under the  GNU GENERAL PUBLIC LICENSE License - see the [LICENSE.md](LICENSE.md) file for details


