-------Data Exploration DataFrames-------

df.head()
df.describe()

#Describing of a single Feature over the binary reslut
df.groupby('Survived').Fare.describe()

#Null Values / Nan
df.isnull().any()
df.isnull().sum()
df.isnull().sum() / df.shape[0]         #in Percentage

#Counting apearance of values of a feature
df.Embarked.value_counts() #/NaN values not included
#Counting the apearance of a specific value of a feature
df[df.Sex == 'female'].Survived.count()


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

#Operations by a group
df.groupby(['Pclass']).Survived.sum()
ag4 = df.groupby('Survived')['Age'].mean()
ag5 = df.groupby('Survived')['Age'].std()





------Data Cleansing-----
# Deleteing a column
df = df.drop(['Cabin'], axis = 1)

