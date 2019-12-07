----DataFrames----
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style="ticks", color_codes=True)
sns.set()

#Counting
sns.countplot(x='Survived', data=df)
#Counting separated with aditional feature (similar to Pivot)
sns.factorplot(x='Survived', col='Sex', kind='count', data=df);

#Percentage in a binary case for a feature
sns.barplot(x='Pclass', y='Survived', data=df)
# similar to this
axes = sns.factorplot('relatives','Survived', 
                      data=df, aspect = 2.5, )


#Histogramm
sns.distplot(df.Fare, kde=False);
#Histogramm combined with a binary result
df.groupby('Survived').Fare.hist(alpha=0.6);

#Plot a feature as a function of the binary result
sns.swarmplot(x='Survived', y='Fare', data=df);

#Pair plot
sns.pairplot(df, hue='Survived');


---#Histogram of a feature with a binary result by group
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = df[df['Sex']=='female']
men = df[df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male')
-------------

----Relationship between two categorical features and a binary result in prercentage
FacetGrid = sns.FacetGrid(df, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()

----Histograms with twi conditional features (1. category, second binary result)
#We create a grid of features and fill it with the histogramm over a 3d feature
grid = sns.FacetGrid(df, col='Survived', row='Pclass', size=3.5, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


---Histograms - several for different features
X_train[['age', 'fare']].hist(bins=30, figsize=(8,4))
plt.show()


-----------COUNTPLOTS

b = sns.countplot(df['G3'])
b.axes.set_title('Distribution of Final grade of students', fontsize = 30)
b.set_xlabel('Final Grade', fontsize = 20)
b.set_ylabel('Count', fontsize = 20)
plt.show()

b = sns.countplot('age', hue='sex', data=df)
b.axes.set_title('Number of students in different age groups',fontsize=30)
b.set_xlabel("Age",fontsize=30)
b.set_ylabel("Count",fontsize=20)
plt.show()



-------------BoxPlot
#shows the distribution e.g. grades(y) for each age group(x)
b = sns.boxplot(x='age', y='G3', data=df)
b.axes.set_title('Age vs Final', fontsize = 30)
b.set_xlabel('Age', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


---------SwarmPlot
# plotting datapoints
b = sns.swarmplot(x='age', y='G3',hue='sex', data=df)
b.axes.set_title('Does age affect final grade?', fontsize = 30)
b.set_xlabel('Age', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()

------Two Distributions of a feature over a feature
# Grade distribution by address
sns.kdeplot(df.loc[df['address'] == 'U', 'G3'], label='Urban', shade = True)
sns.kdeplot(df.loc[df['address'] == 'R', 'G3'], label='Rural', shade = True)
plt.title('Do urban students score higher than rural students?', fontsize = 20)
plt.xlabel('Grade', fontsize = 20);
plt.ylabel('Density', fontsize = 20)
plt.show()


---------Map Correlation between features (and dependent variable) - HEATMAP

import seaborn as sns
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,
        yticklabels=corr.columns)



--------Showing the same graphs for each features
for col in df.columns[1:]:
    ax = sns.countplot(df[col], hue=df['Class'], )
    ax.legend()
    ax.set_title('Counts of mushroom class in '+col)
    plt.show()



----Plotting relationship between dependent variable and an operation of other num features
# let's explore the relationship between the year variables and the house price in a bit of more details
def analyse_year_vars(df, var):
    df = df.copy()
    
    # capture difference between year variable and year the house was sold
    df[var] = df['YrSold'] - df[var]
    
    plt.scatter(df[var], df['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(var)
    plt.show()
    
for var in year_vars:
    if var !='YrSold':
        analyse_year_vars(data, var)


# Checking for Outliers
# with Boxplot
def find_outliers(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()
    
for var in cont_vars:
    find_outliers(data, var)


#Plotting a pivot 
fig = data.groupby(['issue_dt', 'grade'])['loan_amnt'].sum().unstack().plot(
    figsize=(14, 8), linewidth=2)

fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount (US Dollars)')


# Plotting for several features
for var in ['cabin', 'sex', 'embarked']:
    
    fig = plt.figure()
    fig = X_train.groupby([var])['survived'].mean().plot()
    fig.set_title('Relationship between {} and Survival'.format(var))
    fig.set_ylabel('Mean Survival')
    plt.show()
    
    
    
# Distribution of continuous variables - comparisson after scalling
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['AGE'], ax=ax1)
sns.kdeplot(X_train['DIS'], ax=ax1)
sns.kdeplot(X_train['NOX'], ax=ax1)

# after scaling
ax2.set_title('After Standard Scaling')
sns.kdeplot(X_train_scaled['AGE'], ax=ax2)
sns.kdeplot(X_train_scaled['DIS'], ax=ax2)
sns.kdeplot(X_train_scaled['NOX'], ax=ax2)
plt.show()


    






