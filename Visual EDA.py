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






