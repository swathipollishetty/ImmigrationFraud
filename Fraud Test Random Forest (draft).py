#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os  # for OS interface (to get/change directory)
# display and set working/data directory
os.getcwd()
os.chdir('dr')
os.getcwd()


# In[3]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split


# In[4]:


dataset=pd.read_csv('GCMF10k_5.csv')
dataset=dataset.dropna()
dataset=dataset.filter(["Fraud","Years_Married", "Children", "ChildrenBeforeMarriage_Uscitizen",
                        "Previously_Married_Uscitizen", "Previously_Denied_Visa_Immigrant",
                        "Foreign_Residence_Requirement_Immigrant","History_of_Crime_UScitizen",
                        "Expired_Passport_Marriage" ,"HigherEducation_Uscitizen", "Employment_Uscitizen",
                        "Employment_Immigrant","Annual_Income_Immigrant","Annual_Income_Uscitizen",
                        "Sex_Immigrant","Sex_Uscitizen","HigherEducation_Immigrant","Citizenship_Immigrant"])

print(dataset.shape)
print(list(dataset.columns))


# In[5]:


dataset.describe()


# In[6]:


X=dataset.drop('Fraud',axis=1)
y=dataset['Fraud']
X.head()


# from sklearn.preprocessing import LabelEncoder
# Encoder_X = LabelEncoder()
# for col in X.columns:
#     X[col] = Encoder_X.fit_transform(X[col])
# Encoder_y = LabelEncoder()
# y = Encoder_y.fit_transform(y)
# X.head()

# In[7]:


y


# In[8]:


X=pd.get_dummies(X, columns=X.columns, drop_first=True)
X.head()


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[11]:


from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def print_score(classifier, X_train, y_train, X_test, y_test, train=True):
    if train == True:
        print("Training results:\n")
        print('Accuracy Score: {}\n'.format(accuracy_score(y_train,classifier.predict(X_train))))
        print('Classification Report:\n{}\n'.format(classification_report(y_train, classifier.predict(X_train))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train, classifier.predict(X_train))))
        res = cross_val_score(classifier, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')
        print('Average Accuracy:\t{}\n'.format(res.mean()))
        print('Standard Deviation:\t{}'.format(res.std()))
    elif train == False:
        print("Test results:\n")
        print('Accuracy Score: {}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))
        print('Classification Report:\n{}\n'.format(classification_report(y_test, classifier.predict(X_test))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))


# In[12]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 3, criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)


# In[13]:


print_score(classifier, X_train, y_train, X_test, y_test, train=True)


# In[15]:


print_score(classifier, X_train, y_train, X_test, y_test, train=False)


# In[16]:


#Feature importances
classifier.feature_importances_


# In[17]:


rank=classifier.feature_importances_
indices = np.argsort(rank)[::-1]
top_k = 16
new_indices = indices[:top_k]
for i in enumerate(new_indices):
    print(i)


# In[18]:


print("Feature ranking:")
width_graph =[]

for f in range(top_k):
    print("%d. %s (%f)" % (f + 1, X.columns[new_indices[f]], rank[new_indices[f]]))
    width_graph.append(rank[new_indices[f]])


# In[23]:


#Create a feature importance chart
plt.figure()
plt.title("Feature importances")
top_features = X.columns[new_indices]
print(top_features)
plt.barh(top_features[::-1], width_graph[::-1], align="center")
plt.show()


# In[21]:


# ROC_AUC curve and score
from sklearn.metrics import roc_curve
metrics = confusion_matrix(y_train, classifier.predict(X_train))

y_pred_proba = classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba, pos_label = 'Yes')
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="auc score="+str(auc))
plt.legend(loc=4)
plt.show()


# In[22]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1])


# In[64]:




