#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd


# In[41]:


df=pd.read_csv("Iris.csv")


# In[42]:


df.head()


# In[43]:


df.info


# In[44]:


df.describe()


# In[45]:


df.isnull().sum()


# In[46]:


#hence 0 missing values
#hot encoding
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot(x="Species",data=df)


# In[47]:


X=df.drop(['Id'],axis=1)


# In[48]:


X.info


# In[49]:


y=df['Species']


# In[50]:


y.tail()


# In[51]:


X.shape


# In[52]:


sns.heatmap(df.isnull(),yticklabels=False)


# In[53]:


X=pd.get_dummies(X, drop_first=True)


# In[54]:


X.shape


# In[55]:


X.head()


# In[56]:


X.drop('Species_Iris-virginica',inplace=True,axis=1)


# In[57]:


X.drop('Species_Iris-versicolor',inplace=True,axis=1)


# In[58]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state=40)


# In[59]:


X_train.shape


# In[60]:


X_test.shape


# In[61]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)


# In[62]:


#model is ML model for this problem
y_predict=model.predict(X_test)
print(y_predict)


# In[63]:


df1=pd.DataFrame({"Actual": y_test,"Predicted":y_predict })
df1.to_csv("Output_logi_rgr.csv")
df1.head()


# In[64]:


s=model.score(X_test,y_test)
print(s)


# In[65]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[66]:


X.head()


# In[67]:


X.tail()


# In[68]:


df1.head()


# In[95]:


#deployment of model SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
SepalLengthCm=float(input("What is the sepal length:"))
SepalWidthCm=float(input("whatt is the sepal width:"))
PetalLengthCm=float(input("What is the petal length:"))
PetalWidthCm=float(input("What is the petal width:"))
data=[[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]]


# In[96]:


print(data)


# In[97]:


X_train.shape


# In[98]:


y_p=model.predict(data)


# In[99]:


newdf=pd.DataFrame(data, columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
newdf.head()


# In[100]:


newdf=pd.get_dummies(newdf)


# In[101]:


newdf.head()


# In[102]:


y_p=model.predict(newdf)
print(y_p)


# In[ ]:




