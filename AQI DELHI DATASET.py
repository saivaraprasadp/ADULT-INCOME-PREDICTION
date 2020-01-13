#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[97]:


X = pd.read_csv('C:\\Users\\Well\\Downloads\\Train_Combine.csv', usecols=[
                'T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])
X


# In[98]:


Y = pd.read_csv('C:\\Users\\Well\\Downloads\\Train_Combine.csv', usecols=['PM 2.5'])
Y


# In[99]:


X


# In[100]:


Y


# In[101]:


X2 = pd.read_csv('C:\\Users\\Well\\Downloads\\Test_Combine.csv', usecols=[
                 'T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])
X2


# In[102]:


Y2 = pd.read_csv('C:\\Users\\Well\\Downloads\\Test_Combine.csv', usecols=['PM 2.5'])
Y2


# In[103]:


X.isnull().sum()


# In[137]:


X2.isnull().sum()


# In[120]:


Y.info()


# In[149]:


X2.describe().transpose()


# In[107]:


#DT classifier
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(max_depth=5)
model1.fit(X, Y)


# In[108]:


y_pred = model1.predict(X2)
y_pred


# In[109]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y2, y_pred)
cm


# In[110]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y2, y_pred)


# In[111]:


Accuracy_Score = accuracy_score(Y2, y_pred)
Accuracy_Score


# In[112]:


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)  
model2.fit(X,Y)


# In[113]:


y_pred = model2.predict(X2)
y_pred


# In[114]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y2, y_pred)
cm


# In[115]:


Accuracy_Score = accuracy_score(Y2, y_pred)
Accuracy_Score


# In[116]:


#SVM Classifier
from sklearn.svm import SVC
model3 = SVC(kernel = 'rbf', random_state = 0)
model3.fit(X,Y)


# In[117]:


y_pred = model3.predict(X2)
y_pred


# In[118]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y2, y_pred)
cm


# In[119]:


Accuracy_Score = accuracy_score(Y2, y_pred)
Accuracy_Score


# In[125]:


#naiveBayes Classifier
from sklearn.naive_bayes import GaussianNB
model4= GaussianNB()
model4.fit(X ,Y)


# In[126]:


y_pred = model4.predict(X2)
y_pred


# In[127]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y2, y_pred)
cm


# In[128]:


Accuracy_Score = accuracy_score(Y2, y_pred)
Accuracy_Score


# In[129]:


#RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
model5 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
model5.fit(X,Y)


# In[133]:


y_pred = model5.predict(X2)
y_pred


# In[134]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y2, y_pred)
cm


# In[138]:


Accuracy_Score = accuracy_score(Y2, y_pred)
Accuracy_Score


# In[ ]:




