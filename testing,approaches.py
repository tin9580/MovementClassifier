#!/usr/bin/env python
# coding: utf-8

# # Human Movement Classifier

# ### Imports

# In[1]:


from calendar import c
import glob
from operator import index
import os
import pandas as pd
os.chdir('data')
file_extension = '.csv'


# ### Extracting Data

# In[2]:


#files names in each folder
#salsa_files = [i for i in glob.glob(f"salsa/*{file_extension}")]
samba_files = [i for i in glob.glob(f"samba/*{file_extension}")]
walk_files = [i for i in glob.glob(f"walk/*{file_extension}")]
downstairs_files = [i for i in glob.glob(f"downstairs/*{file_extension}")]


# In[5]:


def append_files(files):
    """Function to append multiple files. It returns the files with a column indicating each batch."""
    df_out=pd.DataFrame()
    for i,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        df=df.dropna(axis=1)
        df['batch']=i
        df_out=df_out.append(df,ignore_index = True)
    return(df_out)


# In[6]:


#joining files
#df_salsa = append_files(salsa_files)
df_samba = append_files(samba_files)
df_walk = append_files(walk_files)
df_downstairs = append_files(downstairs_files)


# ### Feature Engineering

# #### 1) Feature Creation - summarizing data into representative values

# Here we will summarize the data considering its standard deviation, maximum value, minimum value, mean, and number of peaks.

# In[7]:


def number_of_peaks(df):
    """return the number of peaks of a signal"""
    from scipy.signal import find_peaks
    peaks,_ = find_peaks(df)
    return len(peaks)


# In[8]:


def create_features(df):
    """for each batch(i.e window of time), we can calculate multiple features, like std, max, min,..."""
    import numpy as np
    df1=df.groupby('batch').agg([np.std, np.max, np.min, np.mean,number_of_peaks])
    #we need to combine the columnames
    df1.columns = df1.columns.map(''.join)
    return(df1.reset_index())


# In[9]:


def max_min(x):
    """Computes the difference between max and min values"""
    return(np.max(x)-np.min(x))

def sum_abs(x):
    """Computes sum of absolute values"""
    return(sum(abs(x)))


# In[10]:


#feature creation
#df_salsa_features = create_features(df_salsa.drop('time',axis=1))
#df_salsa_features['label']='salsa'

df_samba_features = create_features(df_samba.drop('time',axis=1))
df_samba_features['label']='samba'

df_walk_features = create_features(df_walk.drop('time',axis=1))
df_walk_features['label']='walk'

df_downstairs_features = create_features(df_downstairs.drop('time',axis=1))
df_downstairs_features['label']='downstairs'


# In[53]:


#Now we combine everything in a dataset
df = df_samba_features.append(df_walk_features,ignore_index=True)
df = df.append(df_downstairs_features, ignore_index=True)
df=df.drop('batch',axis=1)
df.head(10)


# In[12]:


print("The dataset has {} observations and {} variables".format(df.shape[0], df.shape[1]))


# #### 2) One-hot encoding

# In[14]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[18]:


label = df['label']
label = np.array(label)
label_encoded = LabelEncoder().fit_transform(label)
df_encoded = df.copy()
df_encoded['label']=label_encoded


# ### Exploring Data

# In[21]:


df_walk.plot('time','ax')


# In[22]:


df_downstairs.plot('time','ax')


# In[23]:


df_samba.plot('time','ax')


# ### Preprocessing

# #### Missing values

# In[19]:


df_encoded.isna().sum().sum()


# No missing values!

# #### Outliers

# ### Feature Selection

# #### Correlation Analysis

# In[36]:


def correlation(dataset,target, threshold):
    all_correlation = dataset.corr()
    pos_correlation = all_correlation[all_correlation[target]>threshold]
    neg_correlation = all_correlation[all_correlation[target]<-threshold]
    greatest_correlation = pd.concat([pos_correlation, neg_correlation])
    features = list(greatest_correlation.drop(target).index)
    return features   


# In[38]:


features_selected = correlation(df_encoded, "label", 0.5)


# In[43]:


len(features_selected)


# In[46]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[47]:


#Verify the correlation between the selected variables:
plt.style.use('ggplot')
corr = df_encoded[features_selected].corr()
ax = plt.subplots(figsize=(15, 7))
sns.heatmap(corr,  annot=True, annot_kws={"size": 15}) 


# From the table above, the following are highly correlated. It may be a case of collinearity. This means that they could be representing the same information.
# - gFznumber_of_peaks and aznumber_of_peaks (0.99)
# - gFznumber_of_peaks and wz_number_of_peaks (0.89)
# - aystd and axmean (-0.88)
# - aznumber_of_peaks and wz_number_of_peaks (0.9)
# 
# (and so on - this will change with the data we add)

# ### Modeling

# In[48]:


input_data=df_encoded.drop(['label'],axis=1)
input_data.head()


# In[49]:


X = input_data
y = label_encoded


# In[50]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

# tranformation on the original dataset
X.loc[:,:]=scaler.fit_transform(X)


# In[51]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.33)


# In[52]:


print(f'X_train: {X_train.shape}')
print(f'y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_test: {y_test.shape}')


# ## KNN

# In[54]:


from sklearn.neighbors import KNeighborsClassifier


# In[57]:


#range of k values to check accuracy
kVals = range(1,5)
#empty list to receive accuracies
accuracies = []

for k in kVals:
    
    #training the KNN model with each value of k
    KNN = KNeighborsClassifier(n_neighbors = k)
    KNN.fit(X_train, y_train)
          
    #evaluating the model and updating the list of accuracies
    score = KNN.score(X_test, y_test)
    print("k = %d, accuracy= %.2f%%" % (k, score * 100))
    accuracies.append(score)


# In[58]:


#obtaining the value of k that caused the highest accuracy
i = np.argmax(accuracies)
print("k = %d achieved the highest accuracy of %.2f%%"%(kVals[i], accuracies[i]*100))


# In[61]:


KNN_final = KNeighborsClassifier(n_neighbors = kVals[i]).fit(X_train, y_train)
predictions = KNN_final.predict(X_test)


# #### Model evaluation
# 
# Here I think we can explore a lot

# In[62]:


from sklearn.metrics import classification_report


# In[68]:


print(f"Model Evaluation on Test Data \n {classification_report(y_test, predictions)}")


# In[64]:


from sklearn.metrics import confusion_matrix


# In[67]:


print(f"Confusion matrix \n {confusion_matrix(y_test, predictions)}")


# ### Logistic Regression

# In[69]:


from sklearn.linear_model import LogisticRegression


# In[70]:


regr = LogisticRegression().fit(X_train, y_train)
regr.predict(X_test)


# #### Model evaluation

# ### LDA

# In[72]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[73]:


lda = LDA().fit(X_train, y_train)
lda.predict(X_test)


# #### Model evaluation

# ### Regression Tree

# In[75]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# In[80]:


regr_tree = DecisionTreeClassifier().fit(X_train,y_train)
pred = regr_tree.predict(X_test)


# #### Model evaluation

# In[82]:


from sklearn.metrics import accuracy_score


# In[83]:


accuracy_score(y_test,pred)


# ### Random Forest

# In[77]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[78]:


random_forest = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
random_forest.predict(X_test)


# #### Model evaluation

# In[ ]:




