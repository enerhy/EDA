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
