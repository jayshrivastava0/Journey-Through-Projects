#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#sleep_data = pd.read_csv(r"E:\Chrome Downloads\\02_Sleep.csv")


# In[3]:


#sleep_data.tail(10)


# In[4]:


step_data=pd.read_csv("E:\Chrome Downloads\\01_Steps.csv")


# In[5]:


step_data


# In[6]:


plt.plot(figszie=(10,20))
sns.lineplot(data=step_data, x = step_data['calories'], y = step_data['steps'],color='brown')
#sns.boxplot(step_data['calories'])
sns.lineplot(data=step_data, x = step_data['calories'], y = step_data['distance'])
sns.lineplot(data=step_data, x = step_data['calories'], y = step_data['runDistance'])
plt.show()


# In[7]:


#from sklearn.feature_selection import mutual_info_classif


# In[8]:


Y = step_data['calories']


# In[9]:


step_data.drop('calories',axis=1,inplace=True)


# In[10]:


X = step_data
X


# In[11]:


z = X.date[0].split('-')
vv={'year':z[0],'month':z[1],'day':z[2]}
pd.DataFrame(vv,index=[0,1,2,3])


# In[12]:


aaa=[]
aab=[]
aac=[]
for i in range (-1,2113) :
    i=i+1
    aaa.append(X.date[i].split('-')[0])
    aab.append(X.date[i].split('-')[1])
    aac.append(X.date[i].split('-')[2])
    
    
    
z={'year':aaa,'month':aab,'day':aac}


# In[13]:


ff=pd.DataFrame(data=z)


# In[14]:


ff['year']=ff['year'].astype('int')
ff['month']=ff['month'].astype('int')
ff['day']=ff['day'].astype('int')


# In[15]:


X = pd.concat([X,ff],axis=1)


# In[16]:


X=X.drop(['date'],axis=1)


# In[17]:


#X.dtypes


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


#mutual_info_classif(X, Y, discrete_features=True)


# In[19]:


#imp=pd.DataFrame(mutual_info_classif(X,Y),index=X.columns)
#imp


# In[20]:


#from sklearn.linear_model import LogisticRegression


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[23]:


#lr=LogisticRegression(max_iter=1000)


# In[24]:


#lr.fit(X_train,y_train)


# In[25]:


#pred=lr.predict(X_test)


# In[26]:


#sns.histplot(y_test-pred)


# In[27]:


#lr.score(X_train,y_train)


# In[28]:


#from sklearn.naive_bayes import GaussianNB


# In[29]:


#gnb=GaussianNB()


# In[30]:


#gnb.fit(X_train,y_train)


# In[31]:


#gnb_predict=gnb.predict(X_test)


# In[32]:


#gnb.score(X_test,y_test)


# In[33]:


#from sklearn import svm


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[35]:


#for i in np.arange (0.1,0.5,0.05) :
 #   X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=42)
  #  regr =svm.SVR(kernel='linear')
   # regr.fit(X_train,y_train)
    #regr_pred=regr.predict(x_test)
    #print('score is' + regr.score(X_test,y_test)*100)


# In[ ]:





# In[ ]:





# In[36]:


#regr = svm.SVR(kernel='linear')


# In[37]:


#regr.fit(X_train,y_train)


# In[38]:


#regr_pred=regr.predict(X_test)


# In[39]:


#print(regr.score(X_test,y_test)*100)


# In[40]:


#sns.distplot(regr_pred-y_test)


# In[41]:


from sklearn.ensemble import RandomForestRegressor


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=55)


# In[ ]:





# In[ ]:





# In[43]:


rfg=RandomForestRegressor(n_estimators=160)


# In[44]:


rfg.fit(X_train,y_train)


# In[45]:


new_pred=rfg.predict(X_test)


# In[46]:


rfg.score(X_test,y_test)*100


# In[47]:


#for i in np.arange(0,200,5) :
 #   X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=55)
  #  rfg=RandomForestRegressor(n_estimators=160)
   # rfg.fit(X_train,y_train)
    #print('when i is {} score is {}' .format(rfg.score(X_test,y_test)*100,i))


# In[48]:


sns.distplot(new_pred-y_test)


# In[66]:


sns.scatterplot(new_pred,y_test)


# In[98]:


plt.figure(figsize=(5,5))
plt.scatter(y_test,new_pred,c='black')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(new_pred), max(y_test))
p2 = min(min(new_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'r-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[99]:


#sleep_data


# In[106]:


#\: in sleep_data.start[0]


# In[ ]:




