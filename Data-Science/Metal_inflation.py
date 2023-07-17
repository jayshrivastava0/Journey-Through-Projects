#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data= pd.read_csv(r"E:\Chrome Downloads\\alum_gold_nickel_silver_uran_price_changes.csv")


# In[3]:


data


# In[4]:


data.isna().sum()


# In[5]:


data.dropna(inplace=True)


# In[6]:


data.reset_index()


# In[7]:


data.columns


# In[8]:


year=pd.DataFrame(pd.to_datetime(data['Year'],  format='%Y'))


# In[9]:


def month_converter(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.index(month) + 1


# In[10]:


def month():
    x=[]
    for i in data.Month :
        y=month_converter(i)
        x.append(y)
    return x


# In[11]:


month_list=month()


# In[12]:


month_number=pd.DataFrame(month_list, columns = ['Month'])
month_number


# In[13]:


data


# In[14]:


data.drop('Year',axis=1,inplace=True)


# In[15]:


data.drop('Month',axis=1,inplace=True)


# In[16]:


data


# In[17]:


year


# In[18]:


df=pd.concat([data,year,month_number],axis=1)


# In[19]:


df.reset_index()


# In[20]:


df.isna().sum()


# In[21]:


df.dropna(inplace=True)


# In[22]:


df.reset_index()


# In[ ]:





# In[ ]:





# In[66]:


data.columns


# In[23]:


sns.scatterplot(data['Inflation_rate'],data['Price_gold_infl'],data['Price_gold'])


# In[72]:


plt.figure(figsize=(20,10))
#sns.lineplot(data=data,x='Inflation_rate',y='Price_alum')
#sns.lineplot(data=data,x='Inflation_rate',y='Price_gold')
#sns.lineplot(data=data,x='Inflation_rate',y='Price_nickel')
sns.lineplot(data=data,x='Inflation_rate',y='Price_silver')
sns.lineplot(data=data,x='Inflation_rate',y='Price_uran')
sns.lineplot(data=data,x='Inflation_rate',y='Price_gold_infl')
#plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


int(df.Year.astype(str)[0].split('-')[0])


# In[25]:


df.Year.values[0].astype(str).split('-')[0]


# In[26]:


jj=[]
for i in range(0,333) :
    i=i+1
    x=df.Year.values[i].astype(str).split('-')[0]
    x=int(x)
    jj.append(x)
    
    
years=pd.DataFrame(jj,columns=['Year'])


# In[27]:


years


# In[28]:


df=pd.concat([data,years,month_number],axis=1)


# In[29]:


df.dropna(inplace=True)


# In[30]:


df.reset_index()


# In[31]:


#from sklearn import tree


# In[32]:


from sklearn.ensemble import RandomForestRegressor


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


Y = df['Inflation_rate']


# In[35]:


df.drop('Inflation_rate',axis=1,inplace=True)


# In[36]:


X = df


# In[37]:


X


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[39]:


from sklearn import metrics


# In[40]:


X.shape


# In[122]:


y_train = pd.DataFrame(y_train)


# In[78]:


def model(name):
    name.fit(X_train,y_train)
    print ('training score : {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('prediction are :\n {}'.format(y_prediction))
    print('\n')
    
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score is : {}'.format (r2_score))
    
    print('MAE :', metrics.mean_absolute_error(y_test,y_prediction))
    print('MSE :', metrics.mean_squared_error(y_test,y_prediction))
    print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    


# In[79]:


dt= RandomForestRegressor()


# In[80]:


dt.fit(X_train,y_train)


# In[81]:


predict1 = dt.predict(X_test)


# In[82]:


dt.score(X_train,y_train)*100


# In[85]:


r=np.array(y_test)


# In[86]:


r.reshape(-1,1)


# In[ ]:





# In[ ]:





# In[88]:


#predict2=dt.predict(r)


# In[ ]:





# In[55]:


# Plot the results
plt.figure()
#plt.scatter(X, Y, s=20, edgecolor="black", c="darkorange", label="data")

plt.plot(predict1, color='black')
plt.plot(X_test,color='green')
#plt.plot(X_train)

#plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.show()


# In[64]:


sns.lineplot(predict1, label = 'line 1',color = 'brown')
plt.plot(y_test,label = 'line 2', color = 'black')
plt.plot(figsize=(20,5))
plt.show()


# In[90]:


from sklearn.neighbors import KNeighborsClassifier


# In[92]:


knn = KNeighborsClassifier()


# In[113]:


from sklearn.linear_model import LogisticRegression


# In[111]:


ff=np.array(y_train).reshape(-1,1)


# In[144]:


lr=LogisticRegression(solver='lbfgs',max_iter=10000)


# In[145]:


lr.fit(X_train,encoded)


# In[137]:


y_train.reshape(-1,1)
y_train.shape


# In[139]:


from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)


# In[140]:


encoded


# In[148]:


lr.score(X_train,encoded)*100


# In[154]:


prediction=lr.predict(X_test)


# In[161]:


sns.distplot(y_test-prediction)


# In[ ]:




