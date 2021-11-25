

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



#dataset column names:

col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']


#load training data

df_train = pd.read_csv('mlops/aircraft-engine-run-to-failure/src/data/PM_train.txt', sep = ' ', header=None)




#drop extra space columnn

df_train.drop([26,27], axis=1, inplace=True)
df_train.columns = col_names

df_test = pd.read_csv('mlops/aircraft-engine-run-to-failure/src/data/PM_test.txt', sep = ' ', header=None)

df_test.drop([26,27], axis=1, inplace=True)
df_test.columns = col_names

df_truth = pd.read_csv('mlops/aircraft-engine-run-to-failure/src/data/PM_truth.txt', sep = ' ', header=None)

df_truth=df_truth.drop(columns=1)
df_truth = df_truth.rename(columns = {0: 'ttf'}, inplace = False)


df_truth.reset_index(level=0, inplace=True)
df_truth.columns = ['id', 'ttf']
df_truth['id']=df_truth['id']+1



def prepare_train_data(df_, period):
    df_max_cycle = pd.DataFrame(df_.groupby('id')['cycle'].max())
    df_max_cycle.reset_index(level=0, inplace=True)
    df_max_cycle.columns = ['id', 'last_cycle']

    df_test_in = pd.merge(df_train, df_max_cycle, on='id')
    df_test_in['ttf'] = df_test_in['last_cycle'] - df_test_in['cycle']
    df_test_in=df_test_in.drop(columns="last_cycle")
    #create binary classification label
    df_test_in['label_classification'] = df_test_in['ttf'].apply(lambda x: 1 if x <= period else 0)
    return df_test_in

def prepare_test_data(df_, df_truth_, period):
  df_test_last_cycle = pd.DataFrame(df_.groupby('id')['cycle'].max())
  df_test_last_cycle.reset_index(level=0, inplace=True)
  df_test_last_cycle.columns = ['id', 'last_cycle']

  df_test_in = pd.merge(df_, df_test_last_cycle, on='id')

  

  df_test_in = df_test_in[df_test_in['cycle'] == df_test_in['last_cycle']]

  df_test_in.drop(['last_cycle'], axis=1, inplace=True)
  
  df_test_in.reset_index(drop=True, inplace=True)




  #df_test_in = pd.concat([df_test_in, df_truth], axis=1)
  df_test_in = pd.merge(df_test_in, df_truth, on='id')





  #create binary classification label
  df_test_in['label_classification'] = df_test_in['ttf'].apply(lambda x: 1 if x <= period else 0)
  return df_test_in


period=30
df_train_=prepare_train_data(df_train, period)    
df_test_=prepare_test_data(df_test, df_truth, period=30)

df_train_.to_csv("mlops/aircraft-engine-run-to-failure/src/prepare_data/df_train_.csv")
df_test_.to_csv("mlops/aircraft-engine-run-to-failure/src/prepare_data/df_test_.csv")


print("####### done ###########")