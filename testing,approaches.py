#!/usr/bin/env python
# coding: utf-8

# # Human Movement Classifier

# ### Imports
from calendar import c
import glob
from operator import index
import os
import pandas as pd
os.chdir('data')
file_extension = '.csv'


# ### Extracting Data
#files names in each folder
salsa_files = [i for i in glob.glob(f"salsa/*{file_extension}")]
samba_files = [i for i in glob.glob(f"samba/*{file_extension}")]
walk_files = [i for i in glob.glob(f"walking/*{file_extension}")]
downstairs_files = [i for i in glob.glob(f"downstairs/*{file_extension}")]

def append_files(files):
    """Function to append multiple files. It returns the files with a column indicating each batch."""
    df_out=pd.DataFrame()
    for i,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        df=df.dropna(axis=1)
        df['batch']=i
        df_out=df_out.append(df,ignore_index = True)
    return(df_out)

#joining files
df_salsa = append_files(salsa_files)
df_samba = append_files(samba_files)
df_walk = append_files(walk_files)
df_downstairs = append_files(downstairs_files)


# ### Feature Engineering

# #### 1) Feature Creation - summarizing data into representative values

# Here we will summarize the data considering its standard deviation, maximum value, minimum value, mean, and number of peaks.

def number_of_peaks(df):
    """return the number of peaks of a signal"""
    from scipy.signal import find_peaks
    peaks,_ = find_peaks(df)
    return len(peaks)

def create_features(df):
    """for each batch(i.e window of time), we can calculate multiple features, like std, max, min,..."""
    import numpy as np
    df1=df.groupby('batch').agg([np.std, np.max, np.min, np.mean,number_of_peaks])
    #we need to combine the columnames
    df1.columns = df1.columns.map(''.join)
    return(df1.reset_index())

#feature creation
df_salsa_features = create_features(df_salsa.drop('time',axis=1))
df_salsa_features['label']='salsa'

df_samba_features = create_features(df_samba.drop('time',axis=1))
df_samba_features['label']='samba'

df_walk_features = create_features(df_walk.drop('time',axis=1))
df_walk_features['label']='walk'

df_downstairs_features = create_features(df_downstairs.drop('time',axis=1))
df_downstairs_features['label']='downstairs'

#Now we combine everything in a dataset
df = df_salsa_features.append(df_samba_features,ignore_index=True)
df = df.append(df_walk_features, ignore_index=True)
df = df.append(df_downstairs_features, ignore_index=True)
df= df.drop('batch',axis=1)
df.head(10)

print("The dataset has {} observations and {} variables".format(df.shape[0], df.shape[1]))


# #### 2) One-hot encoding
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label = df['label']
label = np.array(label)
label_encoded = LabelEncoder().fit_transform(label)
df_encoded = df.copy()
df_encoded['label']=label_encoded


# ### Exploring Data

df_salsa.plot('time','ax')

df_walk.plot('time','ax')

df_downstairs.plot('time','ax')

df_samba.plot('time','ax')


# ### Preprocessing

# #### Missing values

df_encoded.isna().sum().sum()

# No missing values!

# #### Outliers

# ### Feature Selection

# #### Correlation Analysis

def correlation(dataset,target, threshold):
    all_correlation = dataset.corr()
    pos_correlation = all_correlation[all_correlation[target]>threshold]
    neg_correlation = all_correlation[all_correlation[target]<-threshold]
    greatest_correlation = pd.concat([pos_correlation, neg_correlation])
    features = list(greatest_correlation.drop(target).index)
    return features   

features_selected = correlation(df_encoded, "label", 0.3)

len(features_selected)

features_selected

import seaborn as sns
import matplotlib.pyplot as plt 

#Verify the correlation between the selected variables:
plt.style.use('ggplot')
corr = df_encoded[features_selected].corr()
ax = plt.subplots(figsize=(15, 7))
sns.heatmap(corr,  annot=True, annot_kws={"size": 15}) 


# It is important to verify the correlation also between variables to detect possible collinearity (when variables represent the same information).

# From the table above, the only 2 variables with high correlation are "aystd" and "ayamax" (0.63). However, it is not strong enough to indicate collinearity so I will keep both variables.

# ### Preparing data for the models

input_data=df_encoded.drop(['label'],axis=1)
input_data.head()

X = input_data[features_selected]
y = label_encoded

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

# tranformation on the original dataset
X.loc[:,:]=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.33)

print(f'X_train: {X_train.shape}')
print(f'y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_test: {y_test.shape}')

# ### Modeling

# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_regr = LogisticRegression().fit(X_train, y_train)

# Linear Determinant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA().fit(X_train, y_train)

# Regression Tree
from sklearn.tree import DecisionTreeClassifier
regr_tree = DecisionTreeClassifier().fit(X_train,y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)

# KNN
from sklearn.neighbors import KNeighborsClassifier

kVals = range(1,5)
accuracies = []
for k in kVals:
    #training the KNN model with each value of k
    KNN_testing = KNeighborsClassifier(n_neighbors = k)
    KNN_testing.fit(X_train, y_train)
    #evaluating the model and updating the list of accuracies
    score = KNN_testing.score(X_test, y_test)
    #print("k = %d, accuracy= %.2f%%" % (k, score * 100))
    accuracies.append(score)
# obtaining the value of k with the highest accuracy
i = np.argmax(accuracies)
KNN = KNeighborsClassifier(n_neighbors = kVals[i]).fit(X_train, y_train)

print("k = %d achieved the highest accuracy of %.2f%%"%(kVals[i], accuracies[i]*100))

# ### Model Evaluation

import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

def evaluation(model):
    predictions=cross_val_predict(model, X_test, y_test)
    skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True)
    plt.show()
    print("Acuracy on test data: %.3f%%" % (metrics.accuracy_score(y_test, predictions) * 100.0))
    
evaluation(KNN)

evaluation(log_regr)

evaluation(lda)

evaluation(regr_tree)

evaluation(random_forest)

