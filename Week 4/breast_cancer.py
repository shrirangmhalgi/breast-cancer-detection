import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def logistic_regression(X_train, Y_train):
    # creating LogisticRegression object
    log_reg = LogisticRegression(random_state=0)

    # training the model
    log_reg.fit(X_train, Y_train)

    # printing the training accuracy
    print(f"logistic regression training accuracy = {log_reg.score(X_train, Y_train)}")

    log_reg_pickle_file = open("logistic_regression_pickle", "wb")
    pickle.dump(log_reg, log_reg_pickle_file)
    return log_reg

def decision_tree(X_train, Y_train):
    # creating DecisionTreeClassifier object
    dec_tree = DecisionTreeClassifier(random_state=0)

    # training the model
    dec_tree.fit(X_train, Y_train)

    # printing the training accuracy
    print(f"decision tree accuracy = {dec_tree.score(X_train, Y_train)}")

    dec_tree_pickle_file = open("decision_tree_pickle", "wb")
    pickle.dump(dec_tree, dec_tree_pickle_file)
    return dec_tree

def random_forest(X_train, Y_train):
    # creating DecisionTreeClassifier object
    ran_for = RandomForestClassifier(random_state=0)

    # training the model
    ran_for.fit(X_train, Y_train)

    # printing the training accuracy
    print(f"random forest training accuracy = {ran_for.score(X_train, Y_train)}")

    ran_for_pickle_file = open("random_forest_pickle", "wb")
    pickle.dump(ran_for, ran_for_pickle_file)
    return ran_for

if __name__ == "__main__":
    df = pd.read_csv("../dataset.csv")

    # Step 1: Open the file in your python notebook, print first 5 rows of the dataset 
    # and mention what are the dependent and independent variables in the Data.
    # 1. printing the first 5 rows of the dataset
    # 2. diagnosis is dependent variable and all the others are independent variables
    print(df.head(5))

    # Step 2: Find the statistical parameters of the Data that you have
    print(df.info())

    # Step 3: Find the shape of the Dataset in hand.
    # There are 569 rows and 33 columns in the dataset 
    print(f"The shape of the dataset is {df.shape}")

    # Step 4: Find missing values from the Dataset.
    # There is 1 extra Unnamed column which has 569 null values
    print(df.isna().sum())

    # removing the missing values
    df = df.loc[:, :'fractal_dimension_worst']

    # Step 5: Find the value count of B(Benign) and M(Malignant) cancer cells in the column "diagnosis"¶
    print(df['diagnosis'].value_counts())

    # Encoding the diagnosis values
    diagnosis_label_encoder = LabelEncoder()

    # encoding the B and M values
    df['diagnosis'] = diagnosis_label_encoder.fit_transform(df['diagnosis'])

    # decoding the values
    df['diagnosis'] = diagnosis_label_encoder.inverse_transform(df['diagnosis'])

    # Step 6: Creating a pairplot and mention the findings
    df1 = df.loc[:, 'id' : 'perimeter_mean']
    sns.pairplot(df1, hue='diagnosis')
    # plt.show()

    # Step 7: Create a correlation matrix and mention strongly, weakly and negatively correlated quantities.
    print(df.corr())

    # Step 8: Create a heatmap of the correlated features (helps in visualization)
    sns.heatmap(df.corr(), annot=True, fmt="0.0%")
    # plt.show()

    # Week 3 Tasks
    # encoding the values
    df['diagnosis'] = diagnosis_label_encoder.fit_transform(df['diagnosis'])

    # Seperating the dependent and independent variables
    X = df.iloc[:, 2:32]
    Y = df.iloc[:, 1]

    # splitting the data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Q1: Now, there is a small homeowrk for statistics wherein you have to read about the parameteres, define them in brief and write about two main types of distributions:
    # 1. Gaussian distribution
    # Gaussian distribution or normal distribution is a symetrical distribution which looks like a bell or simply known as the bell curve.
    # This includes concepts of mean and standard deviation.
    # It kindof gives the average range of the entity.
    # Examples of gaussian distribution can be a persons height.
    # The chances of too short or too long person is less. Where as the chances of person with average height is more.
    # Therefore the graph has more height in the centre and is tapering on the either sides.

    # 2. Binomial distribution
    # Binomial distribution is a symetrical distribution.
    # This includes concepts of probability.
    # It gives the probablity whether the event will occur or not.
    # Examples of binomial distribution are predicting the probability of getting 4 heads when 10 coins are tossed.

    # Differentiate between both as well.
    # Gaussian distribution is based on mean and standard deviations
    # Binomial distribution is based on probabilities

    # scaling the values
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.fit_transform(X_test)

    # There are a number of Machine Learning models available which can be employed to read to meaningful conclusions and selecting the right model depending on a variety of factors such as:
    # 1. The accuracy of the model.
    # 2. The interpretability of the model.
    # 3. The complexity of the model.
    # 4. The scalability of the model.
    # 5. How long does it take to build, train, and test the model?
    # 6. How long does it take to make predictions using the model?
    # 7. Does the model meet the business goal?

    # Q2: Selectively write 5 lines about each of the above three algorithms so that even a rather inexperienced person can understand it alongwith dealing all the technicalities.
    # 1. Logistics regression
    # This is a technique used for traditional statistics as well as machine learning.
    # It predicts whether a value is true or false and not continuous values.
    # It fits an S shaped logistic function. The curve goes from 0 to 1 (0 means False and 1 means True).
    # It can be used to acess what variables are useful to classify the samples.

    # 2. Decision tree classifier 
    # Decision tree asks a question and it follows the path of yes and no until it reaches the final conclusion.
    # We start at the top and work our way down.
    # There are various kinds of decision trees.

    # 3. Random Forest classifier
    # Forest as the name suggests is made up by combining many of the decision trees together.
    
    # Week 4 Tasks
    # 1. Logistic Regression
    log_reg = None
    try:
        log_reg_file = open("logistic_regression_pickle", "rb")
        log_reg = pickle.load(log_reg_file)
    except FileNotFoundError as fnfe:
        log_reg = logistic_regression(X_train, Y_train)
    
    # printing the confusion matrix
    # True Positive
    # False Positive
    # False Negative
    # True Negative
    cm = confusion_matrix(Y_test, log_reg.predict(X_test))
    print(f"Confusion Matrix for Logistic Regression = \n{cm}")

    # printing the accuracy of the model
    true_positive = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_negative = cm[1][1]
    print(f"Accuracy of logistic regression is {(true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)}")

    # 2. Decision Tree
    dec_tree = None
    try:
        dec_tree_file = open("decision_tree_pickle", "rb")
        dec_tree = pickle.load(dec_tree_file)
    except FileNotFoundError as fnfe:
        dec_tree = decision_tree(X_train, Y_train)
    
    # printing the confusion matrix
    # True Positive
    # False Positive
    # False Negative
    # True Negative
    cm = confusion_matrix(Y_test, dec_tree.predict(X_test))
    print(f"Confusion Matrix for Decision Tree Classifier = \n{cm}")

    # printing the accuracy of the model
    true_positive = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_negative = cm[1][1]
    print(f"Accuracy of decision tree is {(true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)}")

    # 3. Random Forest
    ran_for = None
    try:
        ran_for_file = open("random_forest_pickle", "rb")
        ran_for = pickle.load(ran_for_file)
    except FileNotFoundError as fnfe:
        ran_for = random_forest(X_train, Y_train)
    
    # printing the confusion matrix
    # True Positive
    # False Positive
    # False Negative
    # True Negative
    cm = confusion_matrix(Y_test, ran_for.predict(X_test))
    print(f"Confusion Matrix for Random Forest Classifier = \n{cm}")

    # printing the accuracy of the model
    true_positive = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_negative = cm[1][1]
    print(f"Accuracy of random forest is {(true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)}")

    # as random forest takes multiple decision trees into consideration, accuracy is the highest