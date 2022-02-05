#%%
import glob
import os
from turtle import down
from lightgbm import plot_tree
import pandas as pd
import numpy as np
os.chdir('/home/tin/CUAS/Intro ML/Project/data')
file_extension = '.csv'

#files names in each folder
salsa_files = [i for i in glob.glob(f"salsa/*{file_extension}")]
samba_files = [i for i in glob.glob(f"samba/*{file_extension}")]
walk_files = [i for i in glob.glob(f"walking/*{file_extension}")]
downstairs_files = [i for i in glob.glob(f"downstairs/*{file_extension}")]
#%%
def append_files(files):
    """Function to append multiple files. It returns the files with a column indicating each batch."""
    df_out=pd.DataFrame()
    for i,file in enumerate(files):
        print(file)
        df = pd.read_csv(file,header=0, skiprows=range(1, 100), skipfooter=100)##first and last rows, usually noise
        df=df.dropna(axis=1)
        df['batch']=i
        df_out=df_out.append(df,ignore_index = True)
    return(df_out)
# %%
#joining files
df_salsa = append_files(salsa_files)
df_samba = append_files(samba_files)
df_walk = append_files(walk_files)
df_downstairs = append_files(downstairs_files)



#%%
def number_of_peaks(x):
    """return the number of peaks of a signal"""
    from scipy.signal import find_peaks
    peaks,_ = find_peaks(x)
    return len(peaks)

#%%
def max_min(x):
    """Computes the difference between max and min values"""
    return(np.max(x)-np.min(x))

def sum_abs(x):
    """Computes sum of absolute values"""
    return(sum(abs(x)))
# %%
def create_features(df):
    """for each batch(i.e window of time), we can calculate multiple features, like std, max, min,..."""
    df1=df.groupby('batch').agg([np.std, np.max, np.min, np.mean,number_of_peaks,max_min, sum_abs])
    #we need to combine the columnames
    df1.columns = df1.columns.map(''.join)
    return(df1.reset_index())

# %%
#feature creation
df_salsa_features = create_features(df_salsa.drop('time',axis=1))
df_salsa_features['label']='salsa'

df_samba_features = create_features(df_samba.drop('time',axis=1))
df_samba_features['label']='samba'

df_walk_features = create_features(df_walk.drop('time',axis=1))
df_walk_features['label']='walk'

df_downstairs_features = create_features(df_downstairs.drop('time',axis=1))
df_downstairs_features['label']='downstairs'
# %%
#Now we combine everything in a dataset
df = df_salsa_features.append(df_samba_features,ignore_index=True)
df = df.append(df_walk_features, ignore_index=True)
df = df.append(df_downstairs_features, ignore_index=True)
df=df.drop('batch',axis=1)
df
# %%
#preprocessing 
print("The dataset has {} observations and {} variables with {} missing values".format(df.shape[0], df.shape[1], df.isna().sum().sum()) )
#%%
import seaborn as sns
import matplotlib.pyplot as plt
#%%
fig, axes = plt.subplots(3, 3, figsize=(20,10))

sns.kdeplot(data = df, x='gFxstd', hue='label', ax=axes[0,0])
sns.kdeplot(data = df, x='gFystd', hue='label', ax=axes[0,1])
sns.kdeplot(data = df, x='gFzstd', hue='label', ax=axes[0,2])
sns.kdeplot(data = df, x='axstd', hue='label', ax=axes[1,0])
sns.kdeplot(data = df, x='aystd', hue='label', ax=axes[1,1])
sns.kdeplot(data = df, x='azstd', hue='label', ax=axes[1,2])
sns.kdeplot(data = df, x='axmax_min', hue='label', ax=axes[2,0])
sns.kdeplot(data = df, x='aymax_min', hue='label', ax=axes[2,1])
sns.kdeplot(data = df, x='azmax_min', hue='label', ax=axes[2,2])
plt.show()
# %%
def correlation(dataset,target, threshold):
    """returns the features of a data set that has a correlation higher than the threshold with a target feature."""
    all_correlation = dataset.corr()
    pos_correlation = all_correlation[all_correlation[target]>threshold]
    neg_correlation = all_correlation[all_correlation[target]<-threshold]
    greatest_correlation = pd.concat([pos_correlation, neg_correlation])
    features = list(greatest_correlation.drop(target).index)
    return features   

#%%
from sklearn.preprocessing import LabelEncoder

label = df['label']
label = np.array(label)
label_encoded = LabelEncoder().fit_transform(label)
df_encoded = df.copy()
df_encoded['label']=label_encoded
#%%
features_selected = correlation(df_encoded, "label", 0.4)
features_selected
#%%
plt.style.use('ggplot')
corr = df_encoded[features_selected].corr()
ax = plt.subplots(figsize=(15, 7))
sns.heatmap(corr,  annot=True, annot_kws={"size": 15}) 

# %%
#seting training and testing sets
x=df_encoded.drop('label',axis=1)
y=df_encoded[['label']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#%%
#Feature Selection using wraper method
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.tree import DecisionTreeClassifier #we could change the classifier, just as an example
model=sfs(DecisionTreeClassifier(),forward=True,verbose=2,cv=2,n_jobs=-1,scoring='accuracy',k_features= len(x.columns))
model.fit(X_train,y_train)

#plot the scores
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
fig1 = plot_sfs(model.get_metric_dict(), kind='std_dev',figsize=(20,5))

plt.ylim([0.7, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()
#In this case lets take only 4 features
#%%
#Wraper output as a table
subsets_wraper = pd.DataFrame(model.subsets_).transpose()
#the feature names with 4 features...
best_features = list(subsets_wraper.iloc[3]['feature_names'])
best_features
# %%
#accuracy using testing data
from sklearn.metrics import accuracy_score
dt=DecisionTreeClassifier().fit(X_train[best_features],y_train)
y_predict = dt.predict(X_test[best_features])
print(accuracy_score(y_test,y_predict))

# # Modeling

# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_regr = LogisticRegression().fit(X_train[best_features], y_train)

# Linear Determinant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA().fit(X_train[best_features], y_train)

# Regression Tree
from sklearn.tree import DecisionTreeClassifier
regr_tree = DecisionTreeClassifier().fit(X_train[best_features],y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train[best_features], y_train)

# KNN
from sklearn.neighbors import KNeighborsClassifier

kVals = range(1,5)
accuracies = []
for k in kVals:
    #training the KNN model with each value of k
    KNN_testing = KNeighborsClassifier(n_neighbors = k)
    KNN_testing.fit(X_train[best_features], y_train)
    #evaluating the model and updating the list of accuracies
    score = KNN_testing.score(X_test[best_features], y_test)
    #print("k = %d, accuracy= %.2f%%" % (k, score * 100))
    accuracies.append(score)
# obtaining the value of k with the highest accuracy
i = np.argmax(accuracies)
KNN = KNeighborsClassifier(n_neighbors = kVals[i]).fit(X_train[best_features], y_train)

print("k = %d achieved the highest accuracy of %.2f%%"%(kVals[i], accuracies[i]*100))

# # Model Evaluation

import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

def evaluation(model):
    predictions=cross_val_predict(model, X_test[best_features], y_test)
    skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True)
    plt.show()
    print("Acuracy on test data: %.3f%%" % (metrics.accuracy_score(y_test, predictions) * 100.0))
    
evaluation(KNN)

evaluation(log_regr)

evaluation(lda)

evaluation(regr_tree)

evaluation(random_forest)

