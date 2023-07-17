#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('E:\Chrome Downloads\WineQT.csv')


# In[3]:


data.head()


# In[4]:


data=data.drop('Id',axis=1)


# In[5]:


data


# In[6]:


data.quality.unique()


# In[7]:


data.plot(figsize=(20,10))


# In[8]:


data.head(0)


# In[9]:


plt.figure(figsize=(20,10))
sns.lineplot(data=data,x='quality',y='volatile acidity')
sns.lineplot(data=data,x='quality',y='citric acid')
sns.lineplot(data=data,x='quality',y='residual sugar')
sns.lineplot(data=data,x='quality',y='chlorides')
sns.lineplot(data=data,x='quality',y='sulphates')
sns.lineplot(data=data,x='quality',y='alcohol')
plt.legend()


# In[10]:


sns.lineplot(data=data,x='quality',y='free sulfur dioxide')


# In[11]:


sns.lineplot(data=data,x='quality',y='total sulfur dioxide')


# In[12]:


sns.lineplot(data=data,x='quality',y='alcohol')


# In[99]:


data['quality'].value_counts()


# In[50]:


#from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_absolute_error,mean_squared_error, median_absolute_error,confusion_matrix,accuracy_score
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier


# In[14]:


# X and Y split

y=data.quality.values
X=data.drop(columns='quality')


# In[15]:


y.shape


# In[16]:


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.75,random_state=50)


# In[17]:


#lr=LinearRegression()


# In[18]:


#lr.fit(X_train,y_train)


# In[19]:


#print('Score the X_train with y-train is', lr.score(X_train,y_train))


# In[20]:


#y_pred= lr.predict(X_test)


# In[21]:


#sns.distplot(y_pred-y_test)


# In[22]:


#log=LogisticRegression(max_iter=30000,solver='lbfgs')


# In[23]:


#log.fit(X_train,y_train)


# In[24]:


#print('Score the X_train with y-train is', log.score(X_train,y_train))


# In[25]:


#def score():
    #for i in np.arange (0.1,1,0.05):
        
      #  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=i,random_state=50)
     #   lg=LogisticRegression(solver='lbfgs',max_iter=30000)
    #    lg.fit(X_train,y_train)
   #     print('Score the X_train with y-train is', lg.score(X_train,y_train),'when i is',[i])
  #      loglist=[lg.score(X_train,y_train)]
 #       #print(loglist)
#    return max(loglist)


# In[26]:


#score()


# In[ ]:





# In[27]:


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=99)


# In[28]:


#knn=KNeighborsClassifier(n_neighbors=2)


# In[29]:


#knn.fit(X_train,y_train)


# In[30]:


#def knn_score():
    #for i in range (1,10)  :
        #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=99)
        #kn=KNeighborsClassifier(n_neighbors=2)
        #kn.fit(X_train,y_train)
        #print('Score the X_train with y-train is', kn.score(X_train,y_train),'when i is',[i])
        #results=kn.score
        #print(results)
    


# In[31]:


#knn_score()


# In[32]:


#knn.score(X_train,y_train)


# In[33]:


#from sklearn.linear_model import TweedieRegressor


# In[34]:


#reg = TweedieRegressor(power=1, alpha=0.5, link='log')


# In[35]:


#reg.fit(X_train,y_train)


# In[36]:


#reg.score(X_train,y_train)


# In[37]:


#from sklearn.linear_model import PassiveAggressiveRegressor


# In[38]:


#from sklearn.linear_model import SGDClassifier


# In[39]:


#sgdc=SGDClassifier()


# In[40]:


#sgdc.fit(X_train,y_train)


# In[41]:


#sgdc.score(X_train,y_train)


# In[42]:


#from sklearn.linear_model import TheilSenRegressor


# In[43]:


#tsr=TheilSenRegressor( fit_intercept=True, copy_X=True, max_subpopulation=10000.0, n_subsamples=None, max_iter=300, tol=0.001, random_state=None, n_jobs=None, verbose=False)


# In[44]:


#tsr.fit(X_train,y_train)


# In[45]:


#tsr.score(X_train,y_train)


# In[46]:


from  sklearn.gaussian_process import GaussianProcessClassifier


# In[47]:


#def gpr():
 #   for i in np.arange(1,100,2):
  #      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.34,random_state=i)
   #     gpr = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None)
    #    gpr.fit(X_train,y_train)
     #   print('Score the X_train with y-train is', gpr.score(X_train,y_train),'when i is',[i])
   # print(pd.DataFrame(X_train))
        


# In[60]:


from sklearn import metrics


# In[51]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.34,random_state=65)


# In[52]:


gpr = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None)
gpr.fit(X_train,y_train)


# In[53]:


gpr.score(X_train,y_train)


# In[56]:


y_pred = gpr.predict(X_test)


# In[57]:


y_pred


# In[61]:


r2_score=metrics.r2_score(y_test,y_pred)


# In[62]:


plt.plot(x, y, label = "line 1", linestyle="-")
plt.plot(y, x, label = "line 2", linestyle="--")


# In[85]:


plt.plot(y_pred, label = 'line 1',color = 'brown')
plt.plot(y_test,label = 'line 2', color = 'white')
plt.plot(figsize=(20,5))
plt.show()


# In[87]:


hh=y_pred-y_test


# In[91]:


sns.boxplot(hh)


# In[ ]:




