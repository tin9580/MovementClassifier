#%%
import glob
import os
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
        #print(file)
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
import seaborn as sns
import matplotlib.pyplot as plt

#%%
fig, axes = plt.subplots(2, 4, figsize=(20,10), sharey=True)

sns.lineplot(data = df_salsa[df_salsa['batch']==1]['ax'], ax=axes[0,0])
axes[0,0].title.set_text('Salsa')
sns.lineplot(data = df_samba[df_samba['batch']==1]['ax'], ax=axes[0,1])
axes[0,1].title.set_text('Samba')
sns.lineplot(data = df_walk[df_walk['batch']==1]['ax'], ax=axes[0,2])
axes[0,2].title.set_text('Walking')
sns.lineplot(data = df_downstairs[df_downstairs['batch']==1]['ax'], ax=axes[0,3])
axes[0,3].title.set_text('Downstairs')


sns.lineplot(data = df_salsa[df_salsa['batch']==11]['ax'], ax=axes[1,0])
axes[1,0].title.set_text('Salsa')
sns.lineplot(data = df_samba[df_samba['batch']==1]['ax'], ax=axes[1,1])
axes[1,1].title.set_text('Samba')
sns.lineplot(data = df_walk[df_walk['batch']==11]['ax'], ax=axes[1,2])
axes[1,2].title.set_text('Walking')
sns.lineplot(data = df_downstairs[df_downstairs['batch']==11]['ax'], ax=axes[1,3])
axes[1,3].title.set_text('Downstairs')


plt.show()
#%%
def mean_number_of_peaks(x):
    """return the mean number of peaks of a signal"""
    from scipy.signal import find_peaks
    peaks,_ = find_peaks(x)
    return len(peaks)/len(x)

#%%
def max_min(x):
    """Computes the difference between max and min values"""
    return(np.max(x)-np.min(x))

def sum_abs(x):
    """Computes mean of absolute values"""
    return(np.mean(abs(x)))
# %%
def create_features(df):
    """for each batch(i.e window of time), we can calculate multiple features, like std, max, min,..."""
    df1=df.groupby('batch').agg([np.std, np.max, np.min, np.mean, mean_number_of_peaks, max_min, sum_abs])
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
if df.isna().sum().sum() !=0 :
    raise ValueError('The dataframe has missing values')

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
features_selected = correlation(df_encoded, "label", 0.2)
features_selected
#%%
plt.style.use('ggplot')
corr = df_encoded[features_selected].corr()
ax = plt.subplots(figsize=(15, 7))
sns.heatmap(corr,  annot=True, annot_kws={"size": 15}) 

# %%
#seting
x=df_encoded.drop('label',axis=1)
y=df_encoded[['label']]

#%%
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
def feature_selection(model, x, y, cv=4):
    """Plots the accuracy vs the number of features selected for a given model"""
    my_sfs=sfs(model,forward=True,verbose=0,cv=cv,n_jobs=-1,scoring='accuracy',k_features= len(x.columns))
    my_sfs.fit(x,y)

    #plot the scores
    from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
    import matplotlib.pyplot as plt
    fig1 = plot_sfs(my_sfs.get_metric_dict(), kind='std_dev',figsize=(20,5))

    plt.ylim([0.5, 1])
    plt.title(f'Sequential Forward Selection for {model}')
    plt.grid()
    plt.show()

    return my_sfs
#%%
def fit_model(model_wraper, n_features):
    """Computes the accuracy of a model, giving its wrapper model and the number of features"""
    #Wraper output as a table
    subsets_wraper = pd.DataFrame(model_wraper.subsets_).transpose()
    
    model=model_wraper.estimator
    
    #the feature names with n_features features...
    best_features = list(subsets_wraper.iloc[n_features-1]['feature_names'])
    acc = round(subsets_wraper.iloc[n_features-1]['avg_score'],4)

    #plot confussion matrix
    #from scikitplot.metrics import plot_confusion_matrix
    #plot_confusion_matrix(y_true=y_test, y_pred=y_predict, normalize=True, title=f'Confusion Matrix for {model}')
    #plt.show()
    return (acc, model, n_features, (best_features))
#%%
# # Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_fs=feature_selection(DecisionTreeClassifier(),x, y,5)#10 were in my example
#%%
dt_model = fit_model(dt_fs, 8)
#%%
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_fs=feature_selection(LogisticRegression(), x, y)#36
#%%
lr_model = fit_model(lr_fs,13)
#%%
# Linear Determinant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_fs=feature_selection(LDA(), x,y, 4)
#%%
lda_model=fit_model(lda_fs, 22)
#%%
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_fs = feature_selection(RandomForestClassifier(), x, y.values.ravel(),2)
#%%
rfc_model = fit_model(rf_fs, 7)
#%%
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_fs = feature_selection(KNeighborsClassifier(), x, y,5)#22

#%%
knn_model = fit_model(knn_fs, 6)

#%%
#wrap up the results
pd.DataFrame((dt_model, lr_model, lda_model, rfc_model, knn_model), columns=('Accuracy', 'Model', 'Number of Features', 'Features')).sort_values('Accuracy', ascending=False)
#%%
#......... we choose the best model and do hyperparameter tuning.

# sampling part of the data to see the results 
subset = df_encoded.sample(n=100)

x_sample=subset.drop('label',axis=1)
y_sample=subset[['label']]

X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(x_sample, y_sample, test_size=0.33, random_state=42)

def feature_selection_simplified(model, X_train, y_train, cv=2):
    """Just select the features"""
    my_sfs=sfs(model,forward=True,verbose=0,cv=cv,n_jobs=-1,scoring='accuracy',k_features= len(X_train.columns))
    my_sfs.fit(X_train,y_train)
    return my_sfs

# testing with the sample
# Decision Tree
dt_fs_sample=feature_selection_simplified(DecisionTreeClassifier(),X_train_sample, y_train_sample)#10 were in my example
dt_model_sample = fit_model(DecisionTreeClassifier(), dt_fs_sample,X_train_sample, y_train_sample, X_test_sample, y_test_sample, 10)

# Logistic Regression
lr_fs_sample=feature_selection_simplified(LogisticRegression(), X_train_sample, y_train_sample)#36
lr_model_sample = fit_model(LogisticRegression(), lr_fs_sample, X_train_sample, y_train_sample, X_test_sample, y_test_sample, 36)

# Linear Determinant Analysis
lda_fs_sample=feature_selection_simplified(LDA(), X_train_sample, y_train_sample)
lda_model_sample=fit_model(LDA(),lda_fs_sample, X_train_sample, y_train_sample, X_test_sample, y_test_sample, 10)

# Random Forest
rf_fs_sample = feature_selection_simplified(RandomForestClassifier(), X_train_sample, y_train_sample.values.ravel())
rfc_model_sample = fit_model(RandomForestClassifier(), rf_fs_sample, X_train_sample, y_train_sample.values.ravel(), X_test_sample, y_test_sample.values.ravel(), 6)

# KNN
knn_fs_sample = feature_selection_simplified(KNeighborsClassifier(), X_train_sample, y_train_sample)#22
knn_model_sample = fit_model(KNeighborsClassifier(),knn_fs_sample, X_train_sample, y_train_sample, X_test_sample, y_test_sample, 22)

#wrap up the results
pd.DataFrame((dt_model_sample, lr_model_sample, lda_model_sample, rfc_model_sample, knn_model_sample), columns=('Test Accuracy', 'Train Accuracy', 'Model', 'Number of Features', 'Features')).sort_values('Test Accuracy', ascending=False)