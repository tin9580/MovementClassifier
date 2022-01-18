#%%
from calendar import c
import glob
from operator import index
import os
import pandas as pd
os.chdir('/home/tin/CUAS/Intro ML/Project/data')
file_extension = '.csv'

#files names in each folder
walk_files = [i for i in glob.glob(f"walk/*{file_extension}")]
dance_files = [i for i in glob.glob(f"dance/*{file_extension}")]
#%%
def append_files(files):
    """Function to append multiple files. It returns the files with a column indicating each batch."""
    df_out=pd.DataFrame()
    for i,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        df=df.dropna(axis=1)
        df['batch']=i
        df_out=df_out.append(df,ignore_index = True)
    return(df_out)
# %%
#joining files
df_walk = append_files(walk_files)
df_dance = append_files(dance_files)
# %%
df_walk.plot('time','ax')
df_dance.plot('time','ax')

#%%
def number_of_peaks(df):
    """return the number of peaks of a signal"""
    from scipy.signal import find_peaks
    peaks,_ = find_peaks(df)
    return len(peaks)
# %%
def create_features(df):
    """for each batch(i.e window of time), we can calculate multiple features, like std, max, min,..."""
    import numpy as np
    df1=df.groupby('batch').agg([np.std, np.max, np.min, np.mean,number_of_peaks])
    #we need to combine the columnames
    df1.columns = df1.columns.map(''.join)
    return(df1.reset_index())

# %%
df_walk_features = create_features(df_walk.drop('time',axis=1))
df_walk_features['label']='walk'
df_dance_features = create_features(df_dance.drop('time',axis=1))
df_dance_features['label']='dance'

# %%
#Now we combine everything in a dataset
df = df_walk_features.append(df_dance_features,ignore_index=True)
df=df.drop('batch',axis=1)
df
# %%
#........ supervised cluster learning......