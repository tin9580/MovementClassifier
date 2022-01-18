#%%
import glob
import os
import pandas as pd
os.chdir('/home/tin/CUAS/Intro ML/Project/data')
file_extension = '.csv'

#files names in each folder
walk_files = [i for i in glob.glob(f"walk/*{file_extension}")]
dance_files = [i for i in glob.glob(f"dance/*{file_extension}")]


#%%
def append_files(files):
    'Function to append multiple files. It returns the files with a column indicating each batch.'
    df_out=pd.DataFrame()
    for i,file in enumerate(files):
        df = pd.read_csv(file,header=0)
        df=df.dropna(axis=1)
        df['batch']=i
        df_out.append(df)

    return df_out
# %%
df_walk = append_files(walk_files)
df_dance = append_files(dance_files)
# %%
