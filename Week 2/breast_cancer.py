import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
plt.show()

# Step 7: Create a correlation matrix and mention strongly, weakly and negatively correlated quantities.
print(df.corr())

# Step 8: Create a heatmap of the correlated features (helps in visualization)
sns.heatmap(df.corr(), annot=True, fmt="0.0%")
plt.show()
