#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


ps=pd.read_csv('C:\\Users\\Well\\Downloads\\pulsar_stars.csv')
ps


# In[6]:


ps.isnull().sum()


# In[10]:


ps['target_class'].value_counts().plot.bar()


# In[11]:


ps.corr()


# In[17]:


sns.pairplot(ps)


# In[20]:


norm = (ps-ps.min())/(ps.max()-ps.min())
norm


# In[21]:


x= ps.drop(['target_class'],axis = 1)
x


# In[22]:


y= ps['target_class'].values
y


# In[24]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)


# In[29]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)


# In[31]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, lr_pred)
cm


# In[35]:


from sklearn.metrics import accuracy_score
Accuracy_Score = accuracy_score(y_test, lr_pred)
Accuracy_Score


# In[37]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfc.fit(x_train,y_train)


# In[38]:


rfc_pred = rfc.predict(x_test)
rfc_pred


# In[39]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, rfc_pred)
cm


# In[40]:


from sklearn.metrics import accuracy_score
Accuracy_Score = accuracy_score(y_test, rfc_pred)
Accuracy_Score


# In[43]:


from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
nb.fit(x_train ,y_train)


# In[44]:


nb_pred = nb.predict(x_test)
nb_pred


# In[45]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, nb_pred)
cm


# In[46]:


from sklearn.metrics import accuracy_score
Accuracy_Score = accuracy_score(y_test, nb_pred)
Accuracy_Score


# In[49]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =13)
knn.fit(x_train,y_train)
knn_pred = knn.predict(x_test)


# In[50]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, knn_pred)
cm


# In[51]:


from sklearn.metrics import accuracy_score
Accuracy_Score = accuracy_score(y_test, knn_pred)
Accuracy_Score


# In[64]:


from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 0)
svc.fit(x_train,y_train)
svc_pred = svc.predict(x_test)


# In[65]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, svc_pred)
cm


# In[66]:


from sklearn.metrics import accuracy_score
Accuracy_Score = accuracy_score(y_test, svc_pred)
Accuracy_Score


# In[67]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)


# In[68]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, dt_pred)
cm


# In[69]:


from sklearn.metrics import accuracy_score
Accuracy_Score = accuracy_score(y_test, dt_pred)
Accuracy_Score


# In[ ]:




