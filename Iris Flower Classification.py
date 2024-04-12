#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


get_ipython().system('pip install scikit-learn==1.3.0')


# In[6]:


data = pd.read_csv('Iris.csv')


# In[7]:


data.head()


# In[8]:


data = data.drop(columns = ['Id'])
data.head()


# In[9]:


data.describe()


# In[10]:


data.info()


# In[11]:


data['Species'].value_counts()


# In[12]:


data.isnull().sum()


# In[13]:


data['SepalLengthCm'].hist()


# In[14]:


data['SepalWidthCm'].hist()


# In[15]:


data['PetalLengthCm'].hist()


# In[16]:


data['PetalWidthCm'].hist()


# In[17]:


colors =['red','yellow','blue']
species = ['virginica','versicolor','setosa']


# In[18]:


colors = ['red', 'orange', 'blue']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']

for i in range(3):
    x = data[data['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c =colors[i], label=species[i])

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.title('Scatter plot of Sepal Length vs Sepal Width')
plt.show()


# In[19]:


colors = ['red', 'orange', 'blue']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']

for i in range(3):
    x = data[data['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c =colors[i], label=species[i])

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.title('Scatter plot of Sepal Length vs Sepal Width')
plt.show()


# In[20]:


colors = ['red', 'orange', 'blue']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']

for i in range(3):
    x = data[data['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c =colors[i], label=species[i])

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.title('Scatter plot of Sepal Length vs Sepal Width')
plt.show()


# In[21]:


colors = ['red', 'orange', 'blue']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']

for i in range(3):
    x = data[data['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c =colors[i], label=species[i])

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.title('Scatter plot of Sepal Length vs Sepal Width')
plt.show()


# In[22]:


df = data.drop(columns =['Species'])
df.head()


# In[20]:


df.corr()


# In[23]:


corr = df.corr()
fig,ax =plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax,cmap='plasma')


# In[25]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

X = data.drop(columns=['Species'])
Y = data['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) 


# In[23]:


#logistic regression
model = LogisticRegression()
model.fit(x_train,y_train)
print("logistic Regression Accuracy: ",model.score(x_test,y_test)*100)


# In[24]:


#knn
model = KNeighborsClassifier()
model.fit(x_train.values,y_train.values)
print('K-nearest neighbors Accuracy:',model.score(x_test,y_test)*100)


# In[25]:


#Decision Tree
model = DecisionTreeClassifier()
model.fit(x_train.values,y_train.values)
print('Decision Tree Accuracy:',model.score(x_test,y_test)*100)


# In[26]:


import pickle
filename ='saved_model.sav'
pickle.dump(model,open(filename,'wb'))


# In[27]:


import pickle


# In[28]:


filename ='saved_model.sav'
try:
    with open(filename,'wb')as file:
        pickle.dump(model,file)
    print('Model Saved Successfully')
except Exception as e :
    print(f'Error saving the model: {e}')
    


# In[29]:


load_model = pickle.load(open(filename,'rb'))


# In[32]:


x_test.head()


# # Classification

# In[37]:


#Sepal Length: 6.0
#Sepal Width: 2.2
#etal Length: 4.0
#Petal Width: 1.0
load_model.predict([[6.0,2.2,4.0,1.0]])


# In[36]:


load_model.predict([[4.7,3.2,1.3,0.2]])


# In[ ]:




