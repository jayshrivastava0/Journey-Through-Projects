#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("F:\work\data\\2 Year IBM Stock Data.csv")


# In[3]:


data


# In[4]:


y=pd.DataFrame(pd.to_datetime(data['time']))


# In[5]:


xx=[]
yy=[]
zz=[]
for i in range(0,216882) :
    i=i+1
    x1=data.time[i].split()[0].split('/')[0]
    y1=data.time[i].split()[0].split('/')[1]
    z1=data.time[i].split()[0].split('/')[2]
    #print(x)
    xx.append(x1)
    yy.append(y1)
    zz.append(z1)
    
x2=data.time[216882].split()[0].split('/')[0]
xx.extend(x2)

y2=data.time[216882].split()[0].split('/')[1]
yy.extend(y2)

z2=data.time[216882].split()[0].split('/')[2]
zz.append(z2)
    


# In[6]:


date=pd.DataFrame(yy,columns=['Date'])
date=date.astype(int)
month=pd.DataFrame(xx,columns=['Month'])
month=month.astype(int)
year=pd.DataFrame(zz,columns=['Year'])
year=year.astype(int)


# In[7]:


hh=[]
mm=[]
for i in range (0,216882):
    i=i+1
    h = data.time[i].split()[1].split(':')[0]
    m = data.time[i].split()[1].split(':')[1]
    hh.append(h)
    mm.append(m)
    
h1=data.time[216882].split()[1].split(':')[0]
hh.append(h1)
m1=data.time[216882].split()[1].split(':')[1]
mm.append(m1)


# In[8]:


hour=pd.DataFrame(hh,columns=['Hour'])
hour=hour.astype(int)
minute=pd.DataFrame(mm,columns=['Minute'])
minute=minute.astype(int)


# In[9]:


data=pd.concat([data,hour,minute,date,month,year],axis=1)


# In[10]:


time=data['time']


# In[11]:


data.drop('time',axis=1,inplace=True)


# In[12]:


data.dtypes


# In[13]:


from sklearn.cluster import KMeans


# In[14]:


kmeans = KMeans(n_clusters=2, random_state=0)


# In[15]:


kmeans.fit(data)


# In[16]:


kmeans.transform(data)


# In[17]:


plt.plot(kmeans.transform(data))


# In[18]:


plt.plot(data.high)
plt.plot(data.low)
plt.plot(data.volume)


# In[19]:


y=kmeans.fit_predict(data)


# In[20]:


#plt.scatter(y,time)


# In[ ]:





# kmeans.score(data)

# In[ ]:





# In[21]:


#plt.scatter(kmeans.predict(data))


# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


rfc=RandomForestClassifier()


# In[24]:


plt.plot(data.high)


# In[25]:


y=data['high']


# In[26]:


data.drop('high',axis=1,inplace=True)


# In[27]:


data


# In[28]:


x=data.volume


# In[29]:


data.drop('volume',axis=1,inplace=True)


# In[30]:


data


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,y_train,y_test=train_test_split(data,x,train_size=0.75,random_state=100)


# In[33]:


y_train.shape


# In[34]:


from sklearn.naive_bayes import GaussianNB


# In[35]:


gnb=GaussianNB()


# In[36]:


gnb.fit(X_train,y_train)


# In[37]:


#gnb.predict(X_test)


# In[ ]:




