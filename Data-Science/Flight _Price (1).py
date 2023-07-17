#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[ ]:





# In[3]:


train_data = pd.read_excel("E:\Chrome Downloads\Data_Train.xlsx")


# In[ ]:





# In[4]:


train_data = pd.read_excel("E:\Chrome Downloads\Data_Train.xlsx")


# In[5]:


train_data.head()


# In[ ]:





# In[6]:


train_data.isna()


# In[7]:


train_data.isna


# In[8]:


train_data.isna().sum()


# In[9]:


train_data.dropna(inplace=True)


# In[ ]:





# In[ ]:




train_data.isna().sum()
# In[10]:


train_data.isna().sum()


# In[11]:


train_data.dtypes


# In[12]:


train_data.columns


# In[13]:


def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])


# In[14]:


for _ in ['Date_of_Journey','Arrival_Time','Dep_Time']:
    change_into_datetime(_)


# In[15]:


train_data.dtypes


# In[16]:


train_data['Journey_day']=train_data['Date_of_Journey'].dt.day
train_data['Journey_month']=train_data['Date_of_Journey'].dt.month
train_data['Journey_year']=train_data['Date_of_Journey'].dt.year


# In[ ]:





# In[ ]:





# In[17]:


train_data.head()


# In[18]:


train_data.drop('Date_of_Journey',axis=1,inplace=True)


# In[19]:


train_data.head()


# In[20]:


def extract_hour(df,col):
    df[col+'_hour']=df[col].dt.hour

def extract_min(df,col):
    df[col+'_minute'] = df[col].dt.minute
    
def drop_column(df,col):
    df.drop(col,axis=1,inplace=True)


# In[21]:


extract_hour(train_data,'Dep_Time')
extract_min(train_data,'Dep_Time')
drop_column(train_data,'Dep_Time')


# In[22]:


extract_hour(train_data, 'Arrival_Time')
extract_min(train_data, 'Arrival_Time')
drop_column(train_data,'Arrival_Time')


# In[23]:


train_data.head()


# In[24]:


drop_column(train_data,'Additional_Info')

train_data.head()

# In[ ]:





# In[25]:


duration=list(train_data['Duration'])


# In[26]:


for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:
            duration[i] = duration[i]+" 0m"
        if 'm' in duration[i]:
            duration[i] = '0h ' + duration[i]
            


# In[27]:


train_data['Duration'] = duration


# In[28]:


train_data.head()


# In[29]:


def hour(a):
    return a.split(' ')[0][0:-1]
def minute(a):
    return a.split(' ')[1][0:-1]


# In[30]:


train_data['Duration'].apply(hour)


# In[31]:


train_data['Duration_Hour'] = train_data['Duration'].apply(hour)
train_data['Duration_Minute'] = train_data['Duration'].apply(minute)


# In[32]:


drop_column(train_data, 'Duration')


# In[33]:


train_data.dtypes


# In[34]:


train_data['Duration_Hour'] = train_data['Duration_Hour'].astype(int)
train_data['Duration_Minute'] = train_data['Duration_Minute'].astype(int)


# In[35]:


train_data.dtypes


# In[36]:


cat_col = [col for col in train_data.columns if train_data[col].dtype=='O']
cat_col


# In[37]:


cont_col = [col for col in train_data.columns if train_data[col].dtype != 'O']
cont_col


# In[38]:


## nominal data is not in order, so you have to onehot encoding
## nominal data has some kind of order, so you have to do label encoding


# In[39]:


categorical = train_data[cat_col]


# In[40]:


categorical.head()


# In[41]:


categorical['Airline'].value_counts()


# In[42]:


plt.figure(figsize=(15,5))
sns.boxplot(x='Airline', y= 'Price', data=train_data.sort_values('Price',ascending=False)).set_title('Price according to different airlines')
plt.show()


# In[43]:


plt.figure(figsize=(15,5))
sns.boxplot(x='Total_Stops', y = 'Price', data = train_data).set_title('Price according to total number of stops')
plt.show()


# In[44]:


train_data.head()


# In[45]:


Airline=pd.get_dummies(categorical["Airline"],drop_first=True)


# In[46]:


Airline.head()


# In[47]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='Price', y = 'Source', data = train_data.sort_values('Price', ascending = False)).set_title('Price according to the source destination')
plt.show()


# In[48]:


Source = pd.get_dummies(categorical['Source'],drop_first=True)
Source.head()


# In[ ]:





# In[49]:


categorical['Destination'].value_counts()


# In[50]:


plt.figure(figsize=(15,5))
sns.boxplot(x='Destination', y = 'Price', data = train_data.sort_values('Price', ascending=False)).set_title('Price according to destination')
plt.show()


# In[51]:


Destination = pd.get_dummies(categorical['Destination'],drop_first=True)
Destination.head()


# In[52]:


categorical['Route_1']=categorical['Route'].str.split('→').str[0]
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]


# In[53]:


categorical.head()


# In[54]:


drop_column(categorical,'Route')


# In[55]:


categorical.isna().sum()


# In[56]:


categorical.columns


# In[57]:


for i in ['Route_3', 'Route_4', 'Route_5']:
    categorical[i].fillna('None', inplace= True)


# In[58]:


categorical.isnull().sum()


# In[ ]:





# In[ ]:





# In[59]:


for i in categorical.columns:
    print('{} has total {} categories'.format(i, len(categorical[i].value_counts())))


# In[ ]:





# In[60]:


import sklearn


# In[61]:


from sklearn.preprocessing import LabelEncoder


# In[62]:


encoder = LabelEncoder()


# In[63]:


categorical.columns


# In[64]:


for i in ['Route_1', 'Route_2','Route_3', 'Route_4', 'Route_5']:
    categorical[i]=encoder.fit_transform(categorical[i])


# In[65]:


categorical.head()


# In[66]:


categorical['Total_Stops'].unique()


# In[67]:


dict = {'non-stop': 0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[68]:


categorical['Total_Stops']=categorical['Total_Stops'].map(dict)


# In[69]:


categorical.head()


# In[70]:


data_train1=pd.concat([categorical,Airline,Source,Destination,train_data[cont_col]],axis=1)


# In[71]:


data_train1.head()


# In[72]:


pd.set_option('display.max_columns',30)
data_train1.head()


# In[ ]:





# In[73]:


drop_column(data_train1,'Airline')
drop_column(data_train1,'Destination')
drop_column(data_train1,'Source')


# In[74]:


data_train1.head()


# In[ ]:





# In[75]:


def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1).set_title('Price density of various airlines')
    sns.boxplot(df[col],ax=ax2)


# In[76]:


plot(train_data,'Price')
plt.show()


# In[77]:


train_data['Price'] = np.where(train_data['Price']>=40000, train_data['Price'].median(), train_data['Price'])


# In[78]:


#plot(train_data,'Price')
#plt.show()


# In[79]:


X=data_train1.drop('Price',axis=1)
X.head()


# In[80]:


data_train1.shape


# In[81]:


X.shape


# In[82]:


y= data_train1['Price']
y


# In[83]:


from sklearn.feature_selection import mutual_info_classif


# In[ ]:





# In[84]:


X.isna().sum()


# In[85]:


X.fillna(0.00000000001, inplace=True)


# In[ ]:





# In[86]:


X.isna().sum()


# In[87]:


#mutual_info_classif(X,y)


# In[88]:


#imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
#imp


# In[89]:


#imp.columns=['importance']
#imp.sort_values(by='importance',ascending=False)


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)


# In[92]:


from sklearn import metrics
import pickle
import joblib


# In[93]:


def predict(ml_model):
    model=ml_model.fit(X_train,y_train)
    print ('training score : {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('prediction are :\n {}'.format(y_prediction))
    print('\n')
    
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score is : {}'.format (r2_score))
    
    print('MAE :', metrics.mean_absolute_error(y_test,y_prediction))
    print('MSE :', metrics.mean_squared_error(y_test,y_prediction))
    print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    
    sns.distplot(y_test-y_prediction)


# In[94]:


from sklearn.ensemble import RandomForestRegressor


# In[95]:


predict(RandomForestRegressor())
plt.show()


# In[96]:


joblib.dump(predict, 'E:\Jupyterlab\model.pkl')


# In[97]:


from sklearn.linear_model import LinearRegression as lr
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.tree import DecisionTreeClassifier as dtc


# In[98]:


predict(lr())
plt.show()


# In[99]:


predict(dtc())
plt.show()


# In[100]:


##predict(knn())


# In[101]:


from sklearn.ensemble import RandomForestRegressor as reg_rf


# In[102]:


X_train


# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[103]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:





# In[ ]:





# In[104]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
oHe = OneHotEncoder()
ct = ColumnTransformer(transformers=[('encode',oHe,[0])],remainder='passthrough')


# In[105]:


n_estimators=[int(x) for x in np.linspace(start=100,stop=1500,num=6)]
max_depth=[int(x) for x in np.linspace(start=5,stop=1000,num=4)]


# In[106]:


random_grid={
    "n_estimators":[100, 380, 660, 940,1220, 1500],
    "max_features":['auto','sqrt'],
    "max_depth":max_depth,
    "min_samples_split":[5,10,15,100,2000]
}


# In[107]:


Rndscv=RandomizedSearchCV(estimator=reg_rf(),param_distributions=random_grid,cv=2, verbose=2,n_jobs=-1)


# In[108]:


#rndscv.set_params(n_estimators=2000)


# In[109]:


#Rndscv.get_params().keys()


# In[110]:


#rndscv.get_params().keys().estimator__min_samples_split


# In[111]:


Rndscv.fit(X_train,y_train)


# In[112]:


Rndscv.best_params_


# In[113]:


prediction=Rndscv.predict(X_test)


# In[114]:


sns.distplot(y_test-prediction).set_title('Randomized Search CV')
plt.show()


# In[115]:


print((metrics.r2_score(y_test,prediction))*100)


# In[ ]:




