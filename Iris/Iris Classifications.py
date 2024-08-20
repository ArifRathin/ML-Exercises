#!/usr/bin/env python
# coding: utf-8

# In[399]:


from sklearn.datasets import load_iris


# In[400]:


iris_dataset = load_iris()


# In[401]:


print(iris_dataset.keys())


# In[402]:


print(iris_dataset["DESCR"])


# In[403]:


print(iris_dataset["target_names"])


# In[404]:


print(iris_dataset["feature_names"])


# In[405]:


print(iris_dataset["data"].shape)


# In[406]:


print(iris_dataset["target"].shape)


# In[407]:


from sklearn.model_selection import train_test_split


# In[408]:


X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)


# In[409]:


print(X_train.shape)


# In[410]:


print(y_train.shape)


# In[411]:


import pandas as pd


# In[412]:


iris_dataframe = pd.DataFrame(X_train,columns = iris_dataset.feature_names)


# In[413]:


import mglearn


# In[414]:


grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)


# In[415]:


from sklearn.neighbors import KNeighborsClassifier


# In[416]:


knn = KNeighborsClassifier(n_neighbors = 1)


# In[417]:


knn.fit(X_train,y_train)


# In[418]:


import numpy as np


# In[419]:


X_new = np.array([[5.2,9,1,0.2]])


# In[420]:


prediction = knn.predict(X_new)


# In[421]:


print(f"Predicted target name is {iris_dataset['target_names'][prediction]}")


# In[422]:


y_pred = knn.predict(X_test)


# In[423]:


print("Prediction evaluation: {:.2f}".format(np.mean(y_pred == y_test)))


# In[424]:


print("Prediction evaluation using score method of knn object: {:.2f}".format(knn.score(X_test,y_test)))

