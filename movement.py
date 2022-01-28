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
walk_files = [i for i in glob.glob(f"walk/*{file_extension}")]
downstairs_files = [i for i in glob.glob(f"downstairs/*{file_extension}")]
#%%
def append_files(files):
    """Function to append multiple files. It returns the files with a column indicating each batch."""
    df_out=pd.DataFrame()
    for i,file in enumerate(files):
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


# %%
df_samba.plot('time','ax', kind='scatter')
df_salsa.plot('time','gFx', kind='scatter')


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
df.isna().sum().sum()
#...... outliers......... maybe we can do more here....
#we can also do some plots....
# %%
#from https://www.kaggle.com/raviprakash438/wrapper-method-feature-selection
#we can delete this if it is not interesting...
def correlation(dataset,threshold):
    #funcion that tells the corralated columns of a dataset, given a threshold

    col_corr=set() # set will contains unique values.
    corr_matrix=dataset.corr() #finding the correlation between columns.
    for i in range(len(corr_matrix.columns)): #number of columns
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold: #checking the correlation between columns.
                colName=corr_matrix.columns[i] #getting the column name
                col_corr.add(colName) #adding the correlated column name heigher than threshold value.
    return col_corr #returning set of column names
#%%
col=correlation(df.drop('label',axis=1),0.85)
print('Correlated columns:',col)
# %%
#seting training and testing sets
x=df.drop('label',axis=1)
y=df[['label']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#%%
#Feature Selection using wraper method
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.tree import DecisionTreeClassifier #we could change the classifier, just as an example
model=sfs(DecisionTreeClassifier(),forward=True,verbose=2,cv=2,n_jobs=-1,scoring='accuracy',k_features= len(x.columns))
model.fit(X_train,y_train)
# %%
#changing k_Features using the result from above... I got that using only one feature we have a good accuracy, but when we use more data this must change
model=sfs(DecisionTreeClassifier(),forward=True,verbose=2,cv=2,n_jobs=-1,scoring='accuracy',k_features= 1)
model.fit(X_train,y_train)
#Get the column name for the selected feature.
x.columns[list(model.k_feature_idx_)]


# %%
#accuracy using testing data
from sklearn.metrics import accuracy_score
dt=DecisionTreeClassifier().fit(X_train[['gFxamin']],y_train)
y_predict = dt.predict(X_test[['gFxamin']])
print(accuracy_score(y_test,y_predict))
# %%
