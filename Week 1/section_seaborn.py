import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Q7: What are dependent and independent varibles in our dataset?
# dependent variables
# Income per month dependent on Age 
# independent variables
# Name, State, Age, Sex, Number of siblings

df = pd.read_excel(r'../Week 1 Dataset.xlsx') 

# plotting the pairplot
sns.pairplot(df, hue='Number of siblings')
plt.show()

# printing the correlation
print(df.corr(method="pearson"))

# plotting the heatmap
sns.heatmap(df.corr(), annot=True,fmt="0.0%")
plt.show()

# Q8: Mention all your finding from the above given heatmap
# 1. As the age increases, income per month decreases
# 2. As the age increases, number of siblings increases
# 3. As the number of siblings increases, income per month decreases