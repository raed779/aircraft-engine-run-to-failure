


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics 
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle


df_train_ = pd.read_csv('df_train_.csv', sep = ',')
df_train_=df_train_.drop(columns=["Unnamed: 0"])
df_test_ = pd.read_csv('df_test_.csv', sep = ',')
df_test_=df_test_.drop(columns=["Unnamed: 0"])




# original features
original_features= ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

# original + extracted fetures
features_adxf = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9', 'av10', 'av11', 'av12', 'av13', 'av14', 'av15', 'av16', 'av17', 'av18', 'av19', 'av20', 'av21', 'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9', 'sd10', 'sd11', 'sd12', 'sd13', 'sd14', 'sd15', 'sd16', 'sd17', 'sd18', 'sd19', 'sd20', 'sd21']

# features with low or no correlation with regression label
features_lowcr = ['setting3', 's1', 's10', 's18','s19','s16','s5', 'setting1', 'setting2']

# features that have correlation with regression label
features_corrl = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20','s21']


# best :========:>
features=['s12',  's21', 's20', 's6', 's14', 's13', 's3', 's17', 's2', 's15', 's11']


#(s14, s9), (s11, s4), (s12, s7),  (s4, s11), (s8,s13), (s7, s12)

def ScaleMinMax(data_num):
    scaler = MinMaxScaler()
    data_num_sc = pd.DataFrame(scaler.fit_transform(data_num), columns=data_num.columns)
    return data_num_sc


df_train_sc=ScaleMinMax(df_train_.iloc[:,:-2])
df_test_sc=ScaleMinMax(df_test_.iloc[:,:-2])


X_train = df_train_[features]
y_train = df_train_['ttf']

X_test = df_test_[features]
y_test = df_test_['ttf']



#try linear regression

linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)



# Save to file in the current working directory
pkl_filename = "LinearRegression_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(linreg, file)


#try LASSO

lasso = linear_model.Lasso(alpha=0.001)
lasso.fit(X_train, y_train)   
 # Save to file in the current working directory
pkl_filename_lasso = "Lasso_model.pkl"
with open(pkl_filename_lasso, 'wb') as file:
    pickle.dump(lasso, file)


#try Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)


polyreg = linear_model.LinearRegression()
polyreg.fit(X_train_poly, y_train)    
 # Save to file in the current working directory
pkl_filename_polyreg = "polyreg_model.pkl"
with open(pkl_filename_polyreg, 'wb') as file:
    pickle.dump(polyreg, file)


#try Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
#dtrg = DecisionTreeRegressor(max_depth=8, max_features=5, random_state=123) # selected features
dtrg = DecisionTreeRegressor(max_depth=7, random_state=123)
dtrg.fit(X_train, y_train)

 # Save to file in the current working directory
pkl_filename_DecisionTreeRegressor = "DecisionTreeRegressor_model.pkl"
with open(pkl_filename_DecisionTreeRegressor, 'wb') as file:
    pickle.dump(dtrg, file)



#try Random Forest

#rf = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=4, n_jobs=-1, random_state=1) # selected features
rf = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=4, n_jobs=-1, random_state=1) # original features
#rf = RandomForestRegressor(n_estimators=100, max_features=3, max_depth=7, n_jobs=-1, random_state=1) # orig + extrcted 

rf.fit(X_train, y_train)    

 # Save to file in the current working directory
pkl_filename_rf = "RandomForestRegressor.pkl"
with open(pkl_filename_rf, 'wb') as file:
    pickle.dump(rf, file)