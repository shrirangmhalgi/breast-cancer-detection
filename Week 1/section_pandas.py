import pandas as pd

dataset = pd.read_excel(r"../Week 1 Dataset.xlsx")
print(dataset)

# transpose
print(dataset.T)

# seeing top 3 results
print(dataset.head(3))

# seeing bottom 3 results
print(dataset.tail(3))

# seeing the highlevel information
print(dataset.info())

# renaming the columns and saving into new dataset
dataset1 = dataset.rename(columns={'State' : 'State Name'})
print(dataset1)

# renaming the columns and saving into same dataset
dataset.rename(columns={'State' : 'State Name'}, inplace=True)
print(dataset)

# Qn 2 : Rename all the columns in the dataset as per your wish and save it in another dataframe and name it df2
df2 = dataset.rename(columns={'Name' : 'Full Name',
                              'Income per month' : 'Income',
                              'State Name' : 'State',
                              'Age' : 'Number of Years',
                              'Sex' : 'M/F',
                              'Number of siblings' : 'Brothers/Sisters'}) 
print(df2)

# Qn 3 : After you have created df2, make the same changes in the original dataframe using the inplace.
dataset.rename(columns={'Name' : 'Full Name',
                              'Income per month' : 'Income',
                              'State Name' : 'State',
                              'Age' : 'Number of Years',
                              'Sex' : 'M/F',
                              'Number of siblings' : 'Brothers/Sisters'}, inplace=True) 
print(dataset)

# printing specific columns
dataset = pd.read_excel(r"../Week 1 Dataset.xlsx")
print(dataset[['Name', 'Income per month', 'Sex']])

# Q4: Show only the columns information, sex and number of siblings.
print(dataset[['Sex', 'Number of siblings']])

# using iloc
print(dataset.iloc[0:3, 0:2])

# Q5: Use iloc method to print data from column 1 to 4 and rows 3 to 9
print(dataset.iloc[3:9, 1:4])

# finding number of null values
print(dataset.isna().sum()) 

# Basic Statistics with Pandas
print(dataset.describe())

# finding count of individual variables
print(dataset['Sex'].value_counts())

# Q6: Find the total value count of number of siblings in the given dataset
print(dataset['Number of siblings'].value_counts())