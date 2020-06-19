import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

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
# 2. Decision tree classifier 
# 3. Random Forest classifier