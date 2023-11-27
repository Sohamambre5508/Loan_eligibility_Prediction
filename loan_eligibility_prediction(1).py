#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv("loan-train.csv")
dataset.head()


# In[3]:


dataset.shape


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'],margins=True)


# In[7]:


dataset.boxplot(column='ApplicantIncome')


# In[8]:


dataset['ApplicantIncome'].hist(bins=20)


# In[9]:


dataset.boxplot(column='ApplicantIncome',by='Education')


# In[10]:


dataset.boxplot(column='LoanAmount')


# In[11]:


dataset['LoanAmount'].hist(bins=20)


# In[12]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[13]:


dataset.isnull().sum()


# In[14]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[15]:


dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)


# In[16]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[17]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[18]:


dataset.loanAmount=dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log=dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[19]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[20]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[21]:


dataset.isnull().sum()


# In[22]:


dataset['TotalIncome']=dataset["ApplicantIncome"]+dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])


# In[23]:


dataset['TotalIncome_log'].hist(bins=20)


# In[24]:


dataset.head()


# In[25]:


x=dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y=dataset.iloc[:,12].values


# In[26]:


x #independent


# In[27]:


y #dependent


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[29]:


print(x_train)


# In[30]:


from sklearn.preprocessing import LabelEncoder #by using labelEncoder we can convert categorical into numerical(male or female)
labelencoder_x = LabelEncoder()


# In[31]:


for i in range(0,5):
    x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])


# In[32]:


x_train[:,7]=labelencoder_x.fit_transform(x_train[:,7])


# In[33]:


x_train


# In[34]:


labelencoder_y=LabelEncoder()
y_train=labelencoder_y.fit_transform(y_train)


# In[35]:


y_train


# In[36]:


for i in range(0,5):
    x_test[:,i]=labelencoder_x.fit_transform(x_test[:,i])


# In[37]:


x_test[:,7]=labelencoder_x.fit_transform(x_test[:,7])


# In[38]:


labelencoder_y=LabelEncoder()
y_test=labelencoder_y.fit_transform(y_test)


# In[39]:


x_test


# In[40]:


y_test


# In[41]:


from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[42]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier= DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(x_train,y_train)


# In[43]:


y_pred=DTClassifier.predict(x_test)
y_pred


# In[44]:


from sklearn import metrics
print('The accuracy of decision tree is:', metrics.accuracy_score(y_pred,y_test))


# In[45]:


from sklearn.naive_bayes import GaussianNB
NBClassifier= GaussianNB()
NBClassifier.fit(x_train,y_train)


# In[46]:


y_pred=NBClassifier.predict(x_test)


# In[47]:


y_pred


# In[48]:


print('The accuracy of naive Bayes is:',metrics.accuracy_score(y_pred,y_test))


# In[49]:


testdata = pd.read_csv("loan-test.csv")


# In[50]:


testdata.head()


# In[51]:


testdata.info()


# In[52]:


testdata.isnull().sum()


# In[53]:


testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)


# In[54]:


testdata.isnull().sum()


# In[55]:


testdata.boxplot(column='LoanAmount')


# In[56]:


testdata.boxplot(column='ApplicantIncome')


# In[57]:


testdata.LoanAmount= testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[58]:


testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[59]:


testdata.isnull().sum()


# In[60]:


testdata['TotalIncome']=testdata["ApplicantIncome"]+testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])


# In[61]:


testdata.head()


# In[62]:


test=testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[63]:


for i in range(0,5):
    test[:,i]=labelencoder_x.fit_transform(test[:,i])


# In[64]:


test[:,7]=labelencoder_x.fit_transform(test[:,7])


# In[65]:


test


# In[66]:


test=ss.fit_transform(test)


# In[67]:


test


# In[68]:


pred=NBClassifier.predict(test)


# In[69]:


pred # 1=ellgible for the loan and 0= not elligible 


# In[ ]:





# In[ ]:




