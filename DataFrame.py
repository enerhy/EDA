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

#Replacing NaN values 
df["Embarked"].fillna("N", inplace = True)
df['Age'] = df['Age'].fillna(df['Age'].median())

#Adding a column of zeros to the DF
d = pd.DataFrame(0, index=np.arange(418), columns=['category_N'])
df_test = pd.concat([df_test, d], axis=1)

#Rearanging the columns
df_test = df_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'category_female', 'category_male', 'category_C', 'category_N', 'category_Q', 'category_S']]

#Changing data type into numeric
for i in range (len(df.columns) - 1):
    df.iloc[i] = pd.to_numeric(df.iloc[i])


---One Hot Encoder
df['Sex'] = pd.Categorical(df['Sex'])
dfDummies = pd.get_dummies(df['Sex'], prefix = 'category')
df = pd.concat([df, dfDummies], axis=1)
df = df.drop(['Sex'], axis = 1)





