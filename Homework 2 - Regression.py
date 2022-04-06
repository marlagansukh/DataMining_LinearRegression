#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
housing = pd.read_csv(r"C:\Users\marla\Downloads\ames_housing.csv", index_col ='Id',na_values = ['?'])
df = housing[['BldgType', 'OverallQual', 'OverallCond', 'LotFrontage', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GarageCars', 'GarageArea', 'SalePrice']]
print(df.info())
pd.options.mode.chained_assignment = None


# In[272]:


df['LotFrontage'].fillna('0', inplace = True)
print(df.info())


# In[273]:


df.rename(columns = {'1stFlrSF' : 'FirstFlrSF',
                     '2ndFlrSF' : 'SecondFlrSF',
                     '3SsnPorch' : 'ThreeSsnPorch'},
                      inplace = True)
print(df.info())


# In[274]:


df2 = housing.drop(columns = ['BldgType', 'LotFrontage']) #not number 
housing_z = (df - df.mean())/df.std()

housing_z['BldgType']=housing['BldgType']
housing_z['LotFrontage']=housing['LotFrontage']
housing_z.head()


# In[275]:


housing_hot = pd.get_dummies(housing_z, columns = ['BldgType'], drop_first=True)
housing_hot.head()


# In[276]:


housing_hot.to_csv(r"C:\Users\marla\Downloads\working\ames_housing_clean.csv", index = False)


# In[277]:


from sklearn import linear_model 
from sklearn import model_selection as skms 
from sklearn import metrics 
import pandas as pd



# In[278]:


housing_clean = pd.read_csv(r"C:\Users\marla\Downloads\working\ames_housing_clean.csv",na_values = ['?'])
housing_clean.isna().sum()
housing_clean = housing_clean.dropna()
housing_clean.info()


# In[279]:


target = housing_clean['SalePrice']
features = housing_clean.drop(columns = ['SalePrice'])
tts = skms.train_test_split(features,target,
                           test_size=0.25,random_state=42)
(train_ftrs,test_ftrs,train_target,test_target) = tts


# In[280]:


train_ftrs.sort_index().head(10)


# In[281]:


test_ftrs.sort_index().head(10)


# In[282]:


train_target.sort_index().head()


# In[283]:


test_target.sort_index().head()


# In[284]:


lr = linear_model.LinearRegression()
fit = lr.fit(X=train_ftrs, y=train_target)
preds = fit.predict(test_ftrs)
pd.options.display.float_format = '{:.3f}'.format


# In[285]:


mse = metrics.mean_squared_error(test_target,preds).round(3)
print('mean squared error: ', mse)


# In[286]:


from sklearn.feature_selection import RFECV
lr = linear_model.LinearRegression()
rfecv = RFECV(estimator = lr, step = 1,
             scoring = 'neg_mean_squared_error')
rfecv.fit(train_ftrs, train_target)
names = train_ftrs.columns
rank = rfecv.ranking_
excluded = []
for index, name in enumerate(names):
    if rank[index] == 1:
        print(name)
    else: excluded.append(name)

print('\nExcluded Features: ',excluded)


# In[287]:


import seaborn as sns
ax= sns.scatterplot(x= test_target, y=preds, marker='.')
ax.set(xlim=(-3,3),ylim=(-3,3))

