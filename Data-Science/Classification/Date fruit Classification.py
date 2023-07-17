#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_excel("E:\Chrome Downloads\Date_Fruit_Datasets.xlsx")


# In[3]:


data


# In[4]:


# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


le=LabelEncoder()


# In[7]:


y=pd.DataFrame(le.fit_transform(data.Class),columns=['Class'])


# In[8]:


X=data.drop(data.columns[[34]],axis=1)
X


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


rfc = RandomForestClassifier(n_estimators=300)


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)


# In[13]:


X_train.dtypes


# In[ ]:





# In[14]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[15]:


rfc.fit(X_train,np.ravel(y_train))


# In[16]:


np.ravel(y_test)


# In[ ]:





# In[17]:


rfc_pred=rfc.predict(X_test)


# In[18]:


rfc.score(X_test,y_test)


# In[19]:


rfc_prob=rfc.predict_proba(X_test)


# In[ ]:





# In[20]:


from sklearn.tree import DecisionTreeClassifier


# In[21]:


dt=DecisionTreeClassifier()


# In[22]:


dt.fit(X_train,y_train)


# In[23]:


dt_pred=dt.predict(X_test)


# In[24]:


dt.score(X_test,y_test)


# In[25]:


dt_prob=dt.predict_proba(X_test)


# In[ ]:





# In[26]:


from sklearn.naive_bayes import GaussianNB


# In[27]:


gnb = GaussianNB()


# In[28]:


gnb.fit(X_train,np.ravel(y_train))


# In[29]:


gnb.score(X_test,y_test)


# In[30]:


gnb_prob=gnb.predict_proba(X_test)


# In[ ]:





# In[31]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X = sc_X.fit_transform(X)
#sc_X_test = sc_X.fit_transform(X_test)
std=pd.DataFrame(sc_X,columns=[X.columns])
#std_tests = pd.DataFrame(sc_X,columns=[X.columns])


# In[32]:


#rfc.fit(std,np.ravel(y_train))


# In[33]:


std


# In[34]:


#sc_X.fit_transform(y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


#rfc.fit(X_train1,np.ravel(y_train))


# In[36]:


#rfc.score(X_test1,y_test1)*100


# In[37]:


#from sklearn.preprocessing import MinMaxScaler


# In[38]:


#mms=MinMaxScaler()


# In[39]:


#MMS=pd.DataFrame(mms.fit_transform(X),columns=[X.columns])


# In[40]:


#mms_test=mms.fit_transform(y)


# In[42]:


#X_train1,X_test1,y_train1,y_test1=train_test_split(MMS,y,random_state=96)


# In[ ]:





# In[45]:


from sklearn.neighbors import KNeighborsClassifier


# In[46]:


knn=KNeighborsClassifier()


# In[47]:


knn.fit(X_train,np.ravel(y_train))


# In[48]:


knn.score(X_test,y_test)


# In[49]:


#knn.fit(X_train1,np.ravel(y_train1))


# In[ ]:





# In[50]:


from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, dt_prob[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, rfc_prob[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, gnb_prob[:,1], pos_label=1)


# In[51]:


#from sklearn.metrics import roc_auc_score

## auc scores
#auc_score1 = roc_auc_score(y_test, dt_prob[:,1],multi_class='ovr')
#auc_score2 = roc_auc_score(y_test, rfc_prob[:,1],multi_clas='ovr')
#auc_score3 = roc_auc_score(y_test, gnb_prob[:,1],multi_class='ovr')

#print(auc_score1, auc_score2)


# In[52]:


plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Decision Tree')
plt.plot(fpr2, tpr2, linestyle=':',color='green', label='Random Forest')
plt.plot(fpr3, tpr3, linestyle='-',color='green', label='Guassian')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()


# In[53]:


from sklearn.multiclass import OneVsRestClassifier


# In[54]:


clf=OneVsRestClassifier(rfc)


# In[55]:


clf.fit(X_train,y_train)


# In[56]:


ovr_prob=clf.predict_proba(X_test)


# In[57]:


fpr = {}
tpr = {}
thresh ={}

n_class = 7

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, ovr_prob[:,i], pos_label=i)


# In[58]:


plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle=':',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='-.',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='-',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--',color='black', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='-',color='purple', label='Class 5 vs Rest')
plt.plot(fpr[6], tpr[6], linestyle='-.',color='red', label='Class 6 vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);   


# In[59]:


from sklearn.metrics import f1_score


# In[ ]:




