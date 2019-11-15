-------Data Exploration DataFrames-------

df.head()
df.describe()

#Describing of a single Feature over the binary reslut
df.groupby('Survived').Fare.describe()

#Null Values / Nan
df.isnull().any()
df.isnull().sum()
df.isnull().sum() / df.shape[0]         #in Percentage
total = train_df.isnull().sum().sort_values(ascending=False)

#more fancy with percentage
total = data.isnull().sum().sort_values(ascending=False)
percentage = round((data.isnull().sum() / data.shape[0]).sort_values(ascending=False), 2)
missing_data = pd.concat([total, percentage], axis=1, keys=['Total', '%'])
missing_data

#Values counts
ag1 = df['Survived'].value_counts() #aggregates for the values
ag1 = df['Survived'].value_counts().plot(kind = 'bar')

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)



----#Conditional Selection
#Selecting columns with a certain type
df.select_dtypes(include=['object']).info()
#Getting the names of columns with a certain type
categorical_cols = [cname for cname in df.columns if 
                    df[cname].dtype == "object"]

cond1 = df['Age'] > 60 #returns true or falls values for the column
cond2 = df[df['Age'] > 60] #returns the DF with the statisfied condition
#AND
cond4 = df[(df['Age'] == 11) & (df['SibSp'] == 5)] #AND condition
cond5 = df[(df.Age == 11) & (df.SibSp == 5)] #same as previous
#OR
cond6 = df[(df.Age == 11) | (df.SibSp == 5)] #OR
cond7 = df.query('(Age == 11) | (SibSp == 5)') #same as previous
df.loc[:, df.columns != 'b']

#Conditional column
df.loc[df['address'] == 'U', 'G3']


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

--------Correlations between numerical features and a numerical result
df.corr()['G3'].sort_values()


------------Categorical columns:
categorical_cols = [cname for cname in df.columns if 
                    df[cname].dtype == "object"]




------Data Cleansing-----
# Deleteing a column
df = df.drop(['Cabin'], axis = 1)

#Replacing NaN values 
df["Embarked"].fillna("N", inplace = True)
df['Age'] = df['Age'].fillna(df['Age'].median())
#most frequent
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().argmax() )


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

-----------
#AS SIMPLE AS THAT
df5 = pd.get_dummies(df)


category_df = df.select_dtypes(include=['object'])
dummy_df = pd.get_dummies(category_df)




---Alternative converting strings into integers
genders = {"male": 0, "female": 1}
data = [train_df, test_df]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)



#Puting list of prediction into a DataFrame
predict_df = pd.DataFrame(predict, columns=['Survived'])
TEST_PRED = pd.concat([TEST_PRED, predict_df], axis=1)  
TEST_PRED[['PassengerId', 'Survived']].to_csv('Prediction_test.csv', index=False)


---Aggregate 2 coulumns and extract a new column out of them
data = [df, df_test]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
df['not_alone'].value_s()


----Extracting Letters from Feature Values into another column
df1['Deck'] = df1['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
----Extracting Titles from Names
df['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

----Maping of Values to another values
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
dataset['Deck'] = dataset['Deck'].map(deck)

----Replacing string values
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')


-----Cutting Data into categories
Have to check pd.cut // pd.qcut()
----------------------------------


---- Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True
            
#Dropping rows with a certain value
df.drop(df[df.G3 == 0].index, inplace=True)

            
            
------Cutting the dataset to include only feature wich are most correlated (works only for nummerical values)
most_correlated = df5.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]
most_correlated           
 
            

