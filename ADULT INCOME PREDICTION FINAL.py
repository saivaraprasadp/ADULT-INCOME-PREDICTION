#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# Numpy library is used for scientific computing operations & mathematical functions purpose.
# pandas library is used to manipulate and analyse the data.
# matplotlib library is used for data visualization in 2D plotting.
# seaborn library is used for data visualization of high level interface & informative statistical graphs.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing dataset

# importing the data for analysis.

# In[2]:


df_1 = pd.read_csv('C:\\Users\\User\\Downloads\\adult dataset.csv')
df_1


# Removing the unnecessary columns in the dataset.

# In[3]:


df = df_1.drop(['fnlwgt'], axis = 1)
df


# Checking the shape of the dataset  i.e., rows & columns.

# In[4]:


df.shape


# Checking the data types 

# In[5]:


df.info()


# Describe the data is Exploratory Data Analysis.

# In[6]:


df.describe()


# Checking datatype and the column info.

# In[7]:


df.dtypes, df.columns


# Replacing '?' with 'nan' values to find the null values.

# In[8]:


df = df.replace("?", np.nan)
df


# Checking missing values using null function.

# In[9]:


df.isnull().sum()


# Replacing nan values using mode function.

# In[10]:


for col in ['workclass', 'occupation', 'native.country']:
    df[col].fillna(df[col].mode()[0], inplace=True)
    
df    


# In[11]:


df.isnull().sum()


# # Visualization

# # Histogram 

# A histogram is one of the most frequently used data visualization techniques in machine learning. It represents the distribution of a continuous variable over a given interval or period of time. Histograms plot the data by dividing it into intervals called ‘bins’. It is used to inspect the underlying frequency distribution (eg. Normal distribution), outliers, skewness, etc.

# In[70]:


df.hist(bins = 50, figsize= (20,20))
plt.show()


# # Pairplot

# Pairplot is used to understand the best set of features to explain a relationship between two variables or to form the most separated clusters. It also helps to form some simple classification models by drawing some simple lines or make linear separation in our dataset.

# In[ ]:


sns.pairplot(df)


# # Boxplot

# Boxplot is used to check the outliers.

# In[71]:


df.boxplot(figsize = (15,15))
plt.show()


# # Barplot

# A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. The bars can be plotted vertically or horizontally.

# In[72]:


sns.barplot(x="income", y="age", data = df)
plt.show()


# In[73]:


sns.barplot(x="income", y="education", data = df)
plt.show()


# In[74]:


sns.barplot(x="income", y="education.num", data = df)
plt.show()


# In[75]:


sns.barplot(x="income", y="marital.status", data = df)
plt.show()


# In[76]:


sns.barplot(x="income", y="occupation", data = df)
plt.show()


# In[77]:


sns.barplot(x="income", y="relationship", data = df)
plt.show()


# In[78]:


sns.barplot(x="income", y="race", data = df)
plt.show()


# In[79]:


sns.barplot(x="income", y="sex", data = df)
plt.show()


# In[80]:


sns.barplot(x="income", y="native.country", data = df)
plt.show()


# In[81]:


sns.barplot(x="income", y="hours.per.week", data = df)
plt.show()


# # Countplot

# seaborn.countplot is a barplot where the dependent variable is the number of instances of each instance of the independent variable.

# In[83]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['age'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[84]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['workclass'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[85]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['education'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[86]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['education.num'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[87]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['relationship'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[88]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['race'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[89]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['sex'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[90]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['occupation'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[91]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['marital.status'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[92]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['hours.per.week'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# In[93]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['native.country'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# # Distplot

# A distplot plots a univariate distribution of observations. The distplot() function combines the matplotlib hist function with the seaborn kdeplot() and rugplot() functions.

# In[94]:


plt.figure(figsize=(10,5))
sns.distplot(df['income'])
plt.show()


# In[95]:


plt.figure(figsize=(10,5))
sns.distplot(df['age'])
plt.show()


# In[96]:


plt.figure(figsize=(10,5))
sns.distplot(df['education'])
plt.show()


# In[97]:


plt.figure(figsize=(10,5))
sns.distplot(df['education.num'])
plt.show()


# In[98]:


plt.figure(figsize=(10,5))
sns.distplot(df['relationship'])
plt.show()


# In[99]:


plt.figure(figsize=(10,5))
sns.distplot(df['race'])
plt.show()


# In[100]:


plt.figure(figsize=(10,5))
sns.distplot(df['workclass'])
plt.show()


# In[101]:


plt.figure(figsize=(10,5))
sns.distplot(df['hours.per.week'])
plt.show()


# In[102]:


plt.figure(figsize=(10,5))
sns.distplot(df['occupation'])
plt.show()


# In[103]:


plt.figure(figsize=(10,5))
sns.distplot(df['marital.status'])
plt.show()


# In[104]:


plt.figure(figsize=(10,5))
sns.distplot(df['race'])
plt.show()


# In[105]:


plt.figure(figsize=(10,5))
sns.distplot(df['sex'])
plt.show()


# In[106]:


plt.figure(figsize=(10,5))
sns.distplot(df['hours.per.week'])
plt.show()


# In[107]:


plt.figure(figsize=(10,5))
sns.distplot(df['native.country'])
plt.show()


# # kdeplot

# KDE Plot described as Kernel Density Estimate is used for visualizing the Probability Density of a continuous variable. It depicts the probability density at different values in a continuous variable. We can also plot a single graph for multiple samples which helps in more efficient data visualization.

# In[108]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['income'])
plt.show()


# In[109]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['age'])
plt.show()


# In[110]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['education'])
plt.show()


# In[111]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['education.num'])
plt.show()


# In[112]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['workclass'])
plt.show()


# In[113]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['relationship'])
plt.show()


# In[114]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['occupation'])
plt.show()


# In[115]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['marital.status'])
plt.show()


# In[116]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['race'])
plt.show()


# In[117]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['sex'])
plt.show()


# In[118]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['native.country'])
plt.show()


# # Violinplot

# A violin plot is a method of plotting numeric data. It is similar to a box plot, with the addition of a rotated kernel density plot on each side.Typically a violin plot will include all the data that is in a box plot: a marker for the median of the data; a box or marker indicating the interquartile range; and possibly all sample points, if the number of samples is not too high.

# In[119]:


sns.violinplot(x='income',y='age',data=df)


# In[120]:


sns.violinplot(x='income',y='workclass',data=df)


# In[121]:


sns.violinplot(x='income',y='education',data=df)


# In[122]:


sns.violinplot(x='income',y='education.num',data=df)


# In[123]:


sns.violinplot(x='income',y='marital.status',data=df)


# In[124]:


sns.violinplot(x='income',y='occupation',data=df)


# In[125]:


sns.violinplot(x='income',y='relationship',data=df)


# In[126]:


sns.violinplot(x='income',y='race',data=df)


# In[127]:


sns.violinplot(x='income',y='sex',data=df)


# In[128]:


sns.violinplot(x='income',y='native.country',data=df)


# In[129]:


sns.violinplot(x='income',y='hours.per.week',data=df)


# # Jointplot

# Jointplot is seaborn library specific and can be used to quickly visualize and analyze the relationship between two variables and describe their individual distributions on the same plot.

# In[130]:


sns.jointplot(x='income',y='age',data =df, kind = 'hex', gridsize = 20)


# In[131]:


sns.jointplot(x='income',y='workclass',data =df, kind = 'hex', gridsize = 20)


# In[132]:


sns.jointplot(x='income',y='education',data =df, kind = 'hex', gridsize = 20)


# In[133]:


sns.jointplot(x='income',y='education.num',data =df, kind = 'hex', gridsize = 20)


# In[134]:


sns.jointplot(x='income',y='relationship',data =df, kind = 'hex', gridsize = 20)


# In[135]:


sns.jointplot(x='income',y='occupation',data =df, kind = 'hex', gridsize = 20)


# In[136]:


sns.jointplot(x='income',y='marital.status',data =df, kind = 'hex', gridsize = 20)


# In[137]:


sns.jointplot(x='income',y='race',data =df, kind = 'hex', gridsize = 20)


# In[138]:


sns.jointplot(x='income',y='sex',data =df, kind = 'hex', gridsize = 20)


# # Factorplot

# A factor plot is simply the same plot generated for different response and factor variables and arranged on a single page. The underlying plot generated can be any univariate or bivariate plot. The scatter plot is the most common application.

# In[140]:


s = sns.factorplot(x="education.num",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# In[141]:


s = sns.factorplot(x="age",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# In[142]:


s = sns.factorplot(x="workclass",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# In[143]:


s = sns.factorplot(x="education",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# In[144]:


s = sns.factorplot(x="marital.status",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# In[145]:


s = sns.factorplot(x="occupation",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# In[146]:


s = sns.factorplot(x="relationship",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# In[147]:


s = sns.factorplot(x="race",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# In[148]:


s = sns.factorplot(x="sex",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# # Label Encoding 

# Converting categorical variables into numeric format.

# In[25]:


from sklearn.preprocessing import LabelEncoder
lr= LabelEncoder()
df['workclass']=lr.fit_transform(df['workclass'])
df['education']=lr.fit_transform(df['education'])
df['marital.status']=lr.fit_transform(df['marital.status'])
df['occupation']=lr.fit_transform(df['occupation'])
df['relationship']=lr.fit_transform(df['relationship'])
df['race']=lr.fit_transform(df['race'])
df['sex']=lr.fit_transform(df['sex'])
df['income']=lr.fit_transform(df['income'])
df['native.country']=lr.fit_transform(df['native.country'])
df['income']=lr.fit_transform(df['income'])


# In[26]:


df


# In[27]:


df.info()


# # Standardization

# Data standardization is the critical process of bringing data into a common format that allows for collaborative research, large-scale analytics, and sharing of sophisticated tools and methodologies.

# In[28]:


from sklearn.preprocessing import minmax_scale

df[['capital.loss', 'capital.gain']]=minmax_scale(df[['capital.loss','capital.gain']])
df


# # Correlation

# Correlation is used for finding the relationship between the variables.

# In[173]:


df.corr()


# # Heatmap

# A heatmap is a graphical representation of data that uses a system of color-coding to represent different values.

# In[178]:


plt.figure(figsize=(12,12))  
sns.heatmap(df.corr(),cmap='Accent',annot=True)


# # Input & Output variable separation 

# Splitting the independent variable and the target variable.

# In[30]:


x= df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
x.head()

y = df.iloc[:,[13]]
y.head()


# # Splitting the data into Training & Testing Dataset.

# splitting the dataset into training and testing dataset to find the accuracy of the models.

# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[32]:


x_train


# In[33]:


y_train


# In[34]:


x_test


# In[35]:


y_test


# # Importing libraries to find the accuracy.

# Importing libraries to find the confusion matrix, accuracy score.

# In[166]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# # Logistics Regression Classification

# The target variable(or output), y, can take only discrete values for given set of features(or inputs), X.

# In[37]:


#Logistic regression

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)
y_pred =lr.predict(x_test)


# In[38]:


cm = confusion_matrix(y_pred, y_test)
cm


# In[39]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # Decision Tree Classification

# Decision tree uses the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree.

# In[40]:


from sklearn.tree import DecisionTreeClassifier

DTclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTclassifier.fit(x_train, y_train)
y_pred = DTclassifier.predict(x_test)
y_pred


# In[41]:


cm = confusion_matrix(y_pred, y_test)
cm


# In[42]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # K Nearest Neighbour

# KNN classifier stores all the values and classifies new cases by the majority vote by the k neighbour.

# In[43]:


from sklearn.neighbors import KNeighborsClassifier

KNclassifier = KNeighborsClassifier(n_neighbors=5)  
KNclassifier.fit(x_train, y_train)
y_pred = KNclassifier.predict(x_test)
y_pred


# In[44]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[45]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # Naive Bayes Classification

# Bayes Theorem finds the probability of an event occurring given the probability of another event that has already occurred.

# In[46]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred= nb.predict(x_test)
y_pred


# In[47]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[48]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # Random Forest Classification

# Random forest classifier is an esemble classifier made using many decision tree models.

# In[49]:


from sklearn.ensemble import RandomForestClassifier

rfclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfclassifier.fit(x_train, y_train)
y_pred = rfclassifier.predict(x_test)
y_pred


# In[50]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[51]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # Support Vector Machine classification

# In support vector machine we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes very well.

# In[52]:


from sklearn.svm import SVC

SVclassifier = SVC(kernel = 'rbf', random_state = 0)
SVclassifier.fit(x_train, y_train)
y_pred = SVclassifier.predict(x_test)
y_pred


# In[53]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[54]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # AdaBoost Classification

# Boosting algorithms seek to improve the prediction power by training a sequence of weak models, each compensating the weaknesses of its predecessors.

# In[55]:


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_pred


# In[56]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[57]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # XGBoost Classfication

# XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. It is a perfect combination of software and hardware optimization techniques to yield superior results using less computing resources in the shortest amount of time.

# In[58]:


from xgboost import XGBClassifier

XGBclf = XGBClassifier()
XGBclf.fit(x_train, y_train)
y_pred = XGBclf.predict(x_test)
y_pred


# In[59]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[60]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # Bagging Classification

# A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions to form a final prediction.

# In[61]:


from sklearn.ensemble import BaggingClassifier

bclassifier = BaggingClassifier(random_state=1)
bclassifier.fit(x_train,y_train)
y_pred = bclassifier.predict(x_test)
y_pred


# In[62]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[63]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # ExtraTrees Classification

# Is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it’s classification result.

# In[64]:


from sklearn.ensemble import ExtraTreesClassifier

etclassifier = ExtraTreesClassifier(random_state=1)
etclassifier.fit(x_train, y_train)
y_pred = bclassifier.predict(x_test)
y_pred


# In[65]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[66]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # LinearDiscriminantAnalysis

# 
# Discriminant Function Analysis is a dimensionality reduction technique which is commonly used for the supervised classification problems. It is used for modeling differences in groups i.e. separating two or more classes. It is used to project the features in higher dimension space into a lower dimension space.

# In[67]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=5)
lda.fit(x_train,y_train)
y_pred = lda.predict(x_test)
y_pred


# In[68]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[69]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # Conclusion

# The result XGBoost classifier gives the 86.04% prediction comparing to other algorithms.
