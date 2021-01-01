#!/usr/bin/env python
# coding: utf-8

# # Car Dekho Cars Price Prediction

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r'C:\Users\deepak\Desktop\Data Science\Kaggle\Car Dekho Price Prediction\car data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df["Seller_Type"].unique())
print(df["Transmission"].unique())
print(df["Fuel_Type"].unique())


# In[6]:


##check missing or null values


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.columns


# In[10]:


final_dataset= df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[11]:


final_dataset.head()


# In[12]:


final_dataset['Current_Year']=2020


# In[13]:


final_dataset.head()


# In[14]:


final_dataset['no_year']=final_dataset['Current_Year']-final_dataset['Year']


# In[15]:


final_dataset.head()


# In[16]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[17]:


final_dataset.head()


# In[18]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[19]:


final_dataset.head()


# In[20]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[21]:


final_dataset.head()


# In[22]:


final_dataset.corr()


# In[23]:


import seaborn as sns


# In[24]:


sns.pairplot(final_dataset)


# In[25]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))


# In[27]:


g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[28]:


corrmat['Selling_Price'].sort_values(ascending=False)


# In[29]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[30]:


X.head()


# In[31]:


y.head()


# In[32]:


### Feature Importance 
from sklearn.ensemble import ExtraTreesRegressor 
model=ExtraTreesRegressor() 
model.fit(X,y)


# In[33]:


print(model.feature_importances_)


# In[34]:


#plot graph of feature importances for better visualization 
feat_importances = pd.Series(model.feature_importances_, index=X.columns) 
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[37]:


X_train.shape


# In[39]:


y_train.shape


# In[40]:


X_test.shape


# In[41]:


y_test.shape


# In[42]:


from sklearn.ensemble import RandomForestRegressor


# In[43]:


regressor=RandomForestRegressor()


# In[45]:


import numpy as np


# In[46]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[47]:


from sklearn.model_selection import RandomizedSearchCV


# In[48]:



#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[49]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[50]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[51]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[52]:


rf_random.fit(X_train,y_train)


# In[53]:


rf_random.best_params_


# In[54]:


rf_random.best_score_


# In[55]:


predictions=rf_random.predict(X_test)


# In[56]:


sns.distplot(y_test-predictions)


# In[57]:


plt.scatter(y_test,predictions)


# In[ ]:




