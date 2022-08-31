#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as mp
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("E:\Chrome Downloads\smoke_detection_iot.csv")


# In[3]:


data


# In[4]:


data.pop('Unnamed: 0')
data.pop('CNT')


# In[ ]:





# In[5]:


data.isna().sum()


# In[6]:


data.dtypes


# In[ ]:





# In[7]:


data.describe()


# In[8]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot=True)


# In[9]:


y = data['Fire Alarm']
data.drop('Fire Alarm', axis = 1, inplace = True)
x = data


# In[10]:


x = data


# In[60]:





# In[59]:





# In[ ]:





# We will calculate how much different scalers affect the model.
# I have researched about scalers but want to test it out myself.

# In[11]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# In[12]:


min_max = MinMaxScaler()
standard = StandardScaler()
robust = RobustScaler()


# Now the biggest mistake would be to scale the data first and then splitting it.
# That's Why we are splitting the data first and then we would scale it.

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 152)


# In[ ]:





# In[14]:


min_max_training = min_max.fit_transform(X_train)
standard_training = standard.fit_transform(X_train)
robust_training = robust.fit_transform(X_train)


# In[15]:


pd.DataFrame(min_max_training)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


from sklearn.neighbors import KNeighborsClassifier


# In[17]:


knn = KNeighborsClassifier(n_neighbors = 15)


# In[18]:


min_max_knn = knn.fit(min_max_training, y_train)
standard_knn = knn.fit(standard_training, y_train)
robust_knn = knn.fit(robust_training, y_train)
normal_knn = knn.fit(X_train,y_train)


# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


confusion_matrix(y_train, min_max_knn.predict(min_max_training))


# In[21]:


confusion_matrix(y_train, standard_knn.predict(standard_training))


# In[22]:


confusion_matrix(y_train, robust_knn.predict(robust_training))


# In[23]:


confusion_matrix(y_train, normal_knn.predict(X_train))


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


lr = LogisticRegression()


# In[26]:


def lr_model() :
    min_max_lr = lr.fit(min_max_training, y_train)
    standard_lr = lr.fit(standard_training, y_train)
    robust_lr = lr.fit(robust_training, y_train)
    normal_lr = lr.fit(X_train,y_train)
    print( confusion_matrix(y_train, min_max_lr.predict(min_max_training)))
    print( confusion_matrix(y_train, standard_lr.predict(standard_training)))
    print(confusion_matrix(y_train, robust_lr.predict(robust_training)))
    print(confusion_matrix(y_train, normal_lr.predict(X_train)))


# In[27]:


lr_model()


# Fire alarm should trigger on every fire. That means maximum true positives and minimum flase negatives.
# Those other two blocks of false positive and true negative doesn't matter.
# False positive alarms may irate the customer, but if optimizing false postives compromises true positives or accuracy somehow then it's not the way to go.

# In[28]:


from sklearn.naive_bayes import BernoulliNB


# In[29]:


bnb = BernoulliNB()


# In[30]:


bnb.fit(X_train,y_train)


# In[31]:


confusion_matrix(y_train, bnb.predict(min_max_training))


# In[32]:


confusion_matrix(y_train, bnb.predict(standard_training))


# In[33]:


confusion_matrix(y_train, bnb.predict(X_train))


# In[34]:


confusion_matrix(y_train, bnb.predict(robust_training))


# In[35]:


from sklearn.naive_bayes import GaussianNB


# In[36]:


gnb = GaussianNB()


# In[37]:


def gaussian() :
    min_max_gnb = gnb.fit(min_max_training, y_train)
    standard_gnb = gnb.fit(standard_training, y_train)
    robust_gnb = gnb.fit(robust_training, y_train)
    normal_gnb = gnb.fit(X_train,y_train)
    print( confusion_matrix(y_train, gnb.predict(min_max_training))), print('min_max')
    print( confusion_matrix(y_train, gnb.predict(standard_training))), print('standard')
    print( confusion_matrix(y_train, gnb.predict(robust_training))), print('robust')
    print( confusion_matrix(y_train, gnb.predict(X_train))), print('normal')


# In[38]:


gaussian()


# In[39]:


from sklearn.tree import DecisionTreeClassifier


# In[40]:


dtc = DecisionTreeClassifier(max_depth = 10)


# In[41]:


decision_tree = dtc.fit(X_train, y_train)


# In[42]:


confusion_matrix(y_train, dtc.predict(X_train))


# In[ ]:





# In[ ]:





# Damn!!!!!! this confusion matrix ..........., let me check if it isn't overfitting

# In[ ]:





# In[ ]:





# In[43]:


from sklearn import tree


# In[44]:


plt.figure(figsize = (20,20))
tree.plot_tree(decision_tree)


# In[45]:


from sklearn.model_selection import cross_val_score


# In[46]:


scores = cross_val_score(decision_tree, X_train, y_train, cv=5, scoring = 'f1_macro')


# In[47]:


scores


# idk, but it isn't overfitting.

# In[48]:


confusion_matrix(y_test, dtc.predict(X_test))


# In[ ]:





# Wow!!!!

# In[63]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


# ROC and AUC curve for Decision Tree Classifier.

# In[71]:


auc = roc_auc_score(y_test, dtc.predict_proba(X_test)[:, 1])

false_positive_rate, true_positive_rate, thresolds = roc_curve(y_test,dtc.predict_proba(X_test)[:, 1])


# In[73]:


plt.figure(figsize=(10, 8), dpi=100)
plt.axis('scaled')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate)
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# In[ ]:




