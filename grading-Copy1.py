#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


df = pd.read_csv('C:\\Users\\praha\\OneDrive\\Desktop\\kaggle datsets\\student-mat.csv')
df.head()


# In[28]:





# In[30]:


df.info()


# In[31]:


df.describe()


# In[32]:


#outlier detection
X = df[['studytime','failures', 'G1', 'G2','freetime','traveltime','Medu','Fedu','age','health','famrel','goout']]
y=df['G3']
def detect_outlier(df):

  flag_outlier = False

  for feature in df:
    column = df[feature]
    mean = np.mean(column)
    std = np.std(column)
    z_scores = (column - mean) / std
    outliers = np.abs(z_scores) > 3

    n_outliers = sum(outliers)
    if n_outliers > 0:
      print("{} has {} outliers".format(column, n_outliers))
      flag_outlier = True

    if flag_outlier==False:
      print("The dataset has no outliers.")
    
    return None
  
detect_outlier(X)


# In[99]:


z = df[df['G3']<1].index
z1 = df[df['G2']<2].index
z2 = df[df['G1']<1].index
print(y)
df.drop(z,inplace=True)
X.drop(z,inplace=True)
df.drop(z1,inplace=True)
X.drop(z1,inplace=True)
df.drop(z2,inplace=True)
X.drop(z2,inplace=True)
df.head(100)
len(df['G3'])
len(X)


# In[100]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X1 = scaler.fit_transform(X)
print(X1)
s_df = pd.DataFrame(X1, columns=X.columns)
print(s_df)
s_df.head()
len(X1)


# In[101]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant  
 
# the indept variables
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = s_df.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(s_df.values, i)
                          for i in range(len(s_df.columns))]
  
print(vif_data)


# In[37]:


df.isnull().sum()


# In[14]:


sns.pairplot(df)


# In[38]:


correlation = s_df.corr()
correlation


# In[39]:


# Having a look at the correlation matrix

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(s_df.corr(), annot=True, fmt='.1g', cmap="viridis",);


# In[40]:


#graph of age vs failures
plt.figure(figsize=(10,10))
sns.barplot(df['age'],df['failures'],palette='viridis')


# In[78]:


len(y)



# In[102]:


from sklearn.model_selection import train_test_split
#independent Variable
X =s_df[['studytime','failures', 'G1', 'G2','freetime','traveltime','Medu','Fedu','age','health','famrel','goout']]


#dependent variable
y =(df['G3'])


#Train Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state= 42)


# In[103]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize = True)

lr.fit(X_train,y_train)

our_pred = lr.predict(X_test)

print(our_pred)


# In[104]:


plt.figure(figsize= (10,6))
sns.lineplot(y_test,our_pred, err_style="bars")
sns.scatterplot(y_test, our_pred)


# In[105]:


#checking for homoscedasticity of errors
import matplotlib.pyplot as plt
from statistics import mean
residuals = our_pred-y_test # true minus predicted
rem=mean(residuals)
print(rem)
plt.scatter(x=our_pred, y=residuals)
plt.axhline(0)
plt.show()



# In[106]:


from scipy.stats import kstest

#perform Kolmogorov-Smirnov test
kstest(residuals, 'norm')


# In[107]:


from statsmodels.graphics.gofplots import qqplot
qqplot(residuals, line='45')
plt.show()


# In[108]:


from statsmodels.stats.stattools import durbin_watson
gfg = durbin_watson(residuals)
print(gfg)


# In[98]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals)


# In[66]:


from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
#fit regression model
X=sm.add_constant(X)
model = sm.OLS(list(y), X).fit()
# White's test
white_test = het_white(model.resid,  model.model.exog)

#define labels to use for output of White's test
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']

#print results of White's test
print(dict(zip(labels, white_test)))


# In[67]:


sns.regplot(y_test,our_pred,scatter_kws={"color": "black"}, line_kws={"color": "red"})

from sklearn import metrics
from sklearn.metrics import r2_score

print('MSE', metrics.mean_squared_error(y_test,our_pred))
print('R2:', r2_score(y_test , our_pred))


# In[68]:


#check accuracy of our model on the test data
lr.score(X_test, y_test)
#display regression coefficients and R-squared value of model
print(lr.intercept_, lr.coef_)


# In[ ]:




