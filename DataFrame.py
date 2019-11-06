-------Data Exploration DataFrames-------

df.head()
df.describe()

#Null Values / Nan
df.isnull().any()
df.isnull().sum()
df.isnull().sum() / df.shape[0]         #in Percentage

#Counting unique values of a feature
df.Embarked.value_counts() #/NaN values not included

#Stats of values 
df['Age'].min()
df['Age'].max()
df['Age'].mean()
df['Age'].median()

OR

df.Age.mean()

#Pivot Table - Structuring as ordered in the function
ag = df.groupby(['Pclass', 'Survived'])['PassengerId'].count()

#to make it a DataFrame
ag1 = ag.reset_index()




------Data Cleansing-----
# Deleteing a column
df = df.drop(['Cabin'], axis = 1)

