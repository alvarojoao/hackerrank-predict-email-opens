
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np
import csv as csv
import math
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cross_validation import train_test_split
import time
from datetime import date
import datetime
get_ipython().magic(u'matplotlib inline')

def saveFileForSubmission(predicted_lables,custonFileName='submission.csv',customHeader=''):
    result = np.c_[predicted_lables]

    np.savetxt(custonFileName, 
           result.astype(int), 
           delimiter=',', 
           header = customHeader, 
           comments = '', 
           fmt='%u')


# In[2]:

train_df = pd.read_csv('training_dataset.csv/training_dataset.csv', header=0)  
train_df.shape


# In[3]:

def preProcessData(dataframe,train=True):
    mailTypes = ['mail_type_1', 'mail_type_2', 'mail_type_3', 'mail_type_4']
    mailCategories = ['mail_category_1', 'mail_category_10', 'mail_category_11',
                    'mail_category_12', 'mail_category_13', 'mail_category_14',
                    'mail_category_15', 'mail_category_16', 'mail_category_17',
                    'mail_category_18', 'mail_category_2', 'mail_category_3',
                    'mail_category_4', 'mail_category_5', 'mail_category_6',
                    'mail_category_7', 'mail_category_8', 'mail_category_9']
    mail_idRange = range(0,11)
    user_idRange = range(0,11)
    
    if(train):
        dataframe = dataframe.reindex_axis(['opened'] + list([a for a in dataframe.columns if a != 'opened']), axis=1)
        dataframe['opened'] = dataframe.clicked.map( {False: 0, True: 1} ).astype(int)
        dataframe = dataframe.drop(['clicked','unsubscribed','open_time','click_time','unsubscribe_time'], axis=1) 
        
    dataframe.loc[ (dataframe.last_online.notnull()),'last_online']=dataframe['last_online'].dropna().map(lambda x:(datetime.datetime.today()-datetime.datetime.fromtimestamp(x)).days)
    mean_last_online_days = dataframe.last_online.mean()
    dataframe.loc[ (dataframe.last_online.isnull()), 'last_online'] = mean_last_online_days
    
    mode_mail_type = dataframe.mail_type.dropna().mode().values
    dataframe.loc[ (dataframe.mail_type.isnull()), 'mail_type'] = mode_mail_type
    dataFrameMailTypesDiff = np.setdiff1d(mailTypes, np.unique(dataframe.mail_type)) 
    dummiesMail_type =  pd.get_dummies(dataframe.mail_type,prefix='col')
    dataframe = pd.concat([dataframe, dummiesMail_type], axis=1)
    for mailType in map(lambda x:"col_"+x,dataFrameMailTypesDiff):
        dataframe = pd.concat([dataframe,pd.DataFrame({mailType: np.zeros(dataframe.shape[0])})],axis=1)
    
    mode_mail_category = dataframe.mail_category.dropna().mode().values
    dataframe.loc[ (dataframe.mail_category.isnull()), 'mail_category'] = mode_mail_category
    dataFrameMailCategoryDiff = np.setdiff1d(mailCategories, np.unique(dataframe.mail_category)) 
    dummiesMail_category =  pd.get_dummies(dataframe.mail_category,prefix='col')
    dataframe = pd.concat([dataframe, dummiesMail_category], axis=1)
    for mailCategoryItem in map(lambda x:"col_"+x,dataFrameMailCategoryDiff):
        dataframe = pd.concat([dataframe,pd.DataFrame({mailCategoryItem: np.zeros(dataframe.shape[0])})],axis=1)
    
    dataframe['hacker_confirmation'] = dataframe.hacker_confirmation.map( {False: 0, True: 1} ).astype(int)
    
    dataTemp  = dataframe['mail_id'].value_counts()
    hist, edges = np.histogram(dataTemp, bins=[ 1.00000000e+00,2.62150000e+03,5.24200000e+03,
                                               7.86250000e+03,1.04830000e+04,1.31035000e+04,
                                               1.57240000e+04,1.83445000e+04,2.09650000e+04,
                                               2.35855000e+04,2.62060000e+04])
    dataTempDict = dataTemp.to_dict()
    dataframe['mail_id'] = dataframe['mail_id'].map(lambda x: [i for i,v in enumerate(edges) if v<=dataTempDict[x]][-1] )
    dummiesMail_id =  pd.get_dummies(dataframe.mail_id,prefix='mail_id')
    dataframe = pd.concat([dataframe, dummiesMail_id], axis=1)
    
    dataFrameMailIdDiff = np.setdiff1d(mail_idRange, np.unique(dataframe.mail_id)) 
    
    for mailIdItem in map(lambda x:"mail_id_"+str(x),dataFrameMailIdDiff):
        dataframe = pd.concat([dataframe,pd.DataFrame({mailIdItem: np.zeros(dataframe.shape[0])})],axis=1)
    
    dataTemp  = dataframe['user_id'].value_counts()
    hist, edges = np.histogram(dataTemp, bins=[1.,11.5,22.,32.5,43.,53.5,64.,74.5,85.,95.5,106.])
    dataTempDict = dataTemp.to_dict()
    dataframe['user_id'] = dataframe['user_id'].map(lambda x: [i for i,v in enumerate(edges) if v<=dataTempDict[x]][-1] )
    dummiesUser_id =  pd.get_dummies(dataframe.user_id,prefix='user_id')
    dataframe = pd.concat([dataframe, dummiesUser_id], axis=1)
    
    dataFrameUserIdDiff = np.setdiff1d(user_idRange, np.unique(dataframe.user_id)) 
    
    for userIdItem in map(lambda x:"user_id_"+str(x),dataFrameUserIdDiff):
        dataframe = pd.concat([dataframe,pd.DataFrame({userIdItem: np.zeros(dataframe.shape[0])})],axis=1)
    
    dataframe = dataframe.drop(['user_id','mail_id','hacker_created_at','sent_time' ,'hacker_timezone','mail_category','mail_type'], axis=1) 
    
    return dataframe
    


# In[4]:

train_df = preProcessData(train_df)


# In[5]:

train_df.head()


# In[6]:

train_df.columns


# In[7]:

from sklearn import linear_model
from sklearn import tree
from sklearn import svm


train_data = train_df.values
x_train, x_test, y_train, y_test = train_test_split(train_data[0::,1::], train_data[0::,0], 
                            test_size = 0.2, random_state = 0) # Split training/test.

#hipotese = linear_model.LogisticRegression(C=1e5)
hipotese = tree.DecisionTreeClassifier(random_state=0)
#hipotese = svm.SVC()
hipotese.fit(x_train, y_train )


# In[8]:

y_true, y_pred = y_test, hipotese.predict(x_test) # Get our predictions
print(classification_report(y_true, y_pred)) # Classification on each digit


# In[9]:

test_df = pd.read_csv('test_dataset.csv/test_dataset.csv', header=0)  
test_df.shape


# In[10]:

test_df = preProcessData(test_df,False)
test_df.shape


# In[12]:

test_data = test_df.values

y_pred = hipotese.predict(test_data).astype(int)


# In[17]:

saveFileForSubmission(y_pred,'submission.csv',)


# In[ ]:



