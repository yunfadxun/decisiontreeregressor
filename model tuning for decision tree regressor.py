#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import numpy as np
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("income.csv")
df.info()


# In[3]:


for c in df.columns[0:3]:
    plt.figure()
    sns.boxplot(df[c], hue=df[c])
    plt.figure()
    sns.distplot(df[c])


# In[4]:


x= df.drop("income", axis=1)
y= df["income"].values.reshape(-1,1)


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size= 0.7, random_state=44)


# In[6]:


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
cart= DecisionTreeRegressor()
cart.fit(x_train,y_train)
y_pred= cart.predict(x_test)


# In[7]:


from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae
r2_cart= r2_score(y_test,y_pred)
mse_cart= mse(y_test, y_pred)
mae_cart= mae(y_test, y_pred)
print(r2_cart, mse_cart, mae_cart)


# In[8]:


params={"criterion":["absolute_error","squared_error"],
       "max_depth":list(range(1,10)),
       "min_samples_split":list(range(2,10)),
       "max_leaf_nodes":list(range(2,10))}


# In[9]:


from sklearn.model_selection import RandomizedSearchCV
cart_rs= RandomizedSearchCV(DecisionTreeRegressor(), params, cv= 5, n_iter=100, n_jobs=-1)
cart_rs.fit(x_train,y_train)


# In[10]:


cart_rs.best_params_


# In[11]:


new_estimator= cart_rs.best_estimator_
y_pred= new_estimator.predict(x_test)
r2_pred= r2_score(y_test,y_pred)
mse_pred= mse(y_test, y_pred)
mae_pred= mae (y_test,y_pred)
print(r2_pred, mse_pred, mae_pred)


# In[12]:


mse_pred<mse_cart


# In[13]:


df2 = pd.read_csv("housingprice.csv")
df2.info()


# In[14]:


df2["Neighborhood"]=df2["Neighborhood"].replace({"Rural":0,"Suburb":1,"Urban":2})


# In[15]:


#defining x and y
x1= df2.drop("Price", axis=1)
y1= df2["Price"].values.reshape(-1,1)


# In[16]:


columns_to_plot= [i for i in df2.columns[0:6] if i != "Neighborhood"]
for i in columns_to_plot:
    plt.figure()
    sns.distplot(df2[i])


# In[17]:


columns_to_plot= [i for i in df2.columns[0:6] if i != df2.columns[3]]
for i in columns_to_plot:
    plt.figure()
    sns.distplot(df2[i])


# In[18]:


numeric_columns= df.select_dtypes(include="number").columns
for i in df[numeric_columns]:
    plt.figure()
    sns.boxplot(df[i])


# In[19]:


numeric_columns


# In[20]:


x1_train, x1_test, y1_train, y1_test= train_test_split(x1,y1, train_size=0.70, random_state=44)
price_predict= DecisionTreeRegressor()
price_predict.fit(x1_train, y1_train)
y_price_predict= price_predict.predict(x1_test)


# In[21]:


print(r2_score(y1_test, y_price_predict), mse(y1_test,y_price_predict), mae(y1_test,y_price_predict))


# In[22]:


params1={"criterion":["absolute_error","squared_error"],
        "min_samples_split":list(range(2,21)),
        "max_depth":list(range(5,10)),
        "max_leaf_nodes":list(range(10,21))}


# In[23]:


price_tuning= RandomizedSearchCV(DecisionTreeRegressor(), params1, cv=5, n_iter=25, n_jobs=-1)
price_tuning.fit(x1_train,y1_train)


# In[24]:


price_tuning.best_params_


# In[28]:


model_price= price_tuning.best_estimator_
y_model_tuned= model_price.predict(x1_test)
print(r2_score(y1_test,y_model_tuned), mse(y1_test, y_model_tuned), mae(y1_test,y_model_tuned))


# In[29]:


mse(y1_test,y_model_tuned)<mse(y1_test,y_price_predict)


# In[ ]:




