
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

from sklearn.preprocessing import PolynomialFeatures




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


X_train = df_train_[features]
y_train = df_train_['ttf']

X_test = df_test_[features]
y_test = df_test_['ttf']


pkl_filename = "LinearRegression_model.pkl"
pkl_filename_polyreg = "polyreg_model.pkl"
pkl_filename_lasso = "Lasso_model.pkl"
pkl_filename_DecisionTreeRegressor = "DecisionTreeRegressor_model.pkl"
pkl_filename_rf = "RandomForestRegressor.pkl"

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

y_test_predict = pickle_model.predict(X_test)
y_train_predict = pickle_model.predict(X_train)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))


regr_metrics = {
                    'Root Mean Squared Error' : metrics.mean_squared_error(y_test, y_test_predict)**0.5,
                    'Mean Absolute Error' : metrics.mean_absolute_error(y_test, y_test_predict),
                    'R^2' : metrics.r2_score(y_test, y_test_predict),
                    'Explained Variance' : metrics.explained_variance_score(y_test, y_test_predict)
                }

#return reg_metrics
df_regr_metrics = pd.DataFrame.from_dict(regr_metrics, orient='index')
df_regr_metrics.columns = ['Linear Regression']


# Load from file
with open(pkl_filename_lasso, 'rb') as file:
    pickle_model_lass = pickle.load(file)

y_test_predict = pickle_model_lass.predict(X_test)
y_train_predict = pickle_model_lass.predict(X_train)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))


lasso_metrics = {
                    'Root Mean Squared Error' : metrics.mean_squared_error(y_test, y_test_predict)**0.5,
                    'Mean Absolute Error' : metrics.mean_absolute_error(y_test, y_test_predict),
                    'R^2' : metrics.r2_score(y_test, y_test_predict),
                    'Explained Variance' : metrics.explained_variance_score(y_test, y_test_predict)
                }

#return reg_metrics
lasso_metrics = pd.DataFrame.from_dict(lasso_metrics, orient='index')
lasso_metrics.columns = ['LASSO']



# Load from file
with open(pkl_filename_polyreg, 'rb') as file:
    pickle_model_polyreg = pickle.load(file)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

y_test_predict = pickle_model_polyreg.predict(X_test_poly)
y_train_predict = pickle_model_polyreg.predict(X_train_poly)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))



polyreg_metrics = {
                    'Root Mean Squared Error' : metrics.mean_squared_error(y_test, y_test_predict)**0.5,
                    'Mean Absolute Error' : metrics.mean_absolute_error(y_test, y_test_predict),
                    'R^2' : metrics.r2_score(y_test, y_test_predict),
                    'Explained Variance' : metrics.explained_variance_score(y_test, y_test_predict)
                }

#return reg_metrics
polyreg_metrics = pd.DataFrame.from_dict(polyreg_metrics, orient='index')
polyreg_metrics.columns = ['Polynomial Regression']

# Load from file
with open(pkl_filename_DecisionTreeRegressor, 'rb') as file:
    pickle_model_DecisionTreeRegressor = pickle.load(file)


y_test_predict = pickle_model_DecisionTreeRegressor.predict(X_test)
y_train_predict = pickle_model_DecisionTreeRegressor.predict(X_train)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))


dtrg_metrics = {
                    'Root Mean Squared Error' : metrics.mean_squared_error(y_test, y_test_predict)**0.5,
                    'Mean Absolute Error' : metrics.mean_absolute_error(y_test, y_test_predict),
                    'R^2' : metrics.r2_score(y_test, y_test_predict),
                    'Explained Variance' : metrics.explained_variance_score(y_test, y_test_predict)
                }

#return reg_metrics
dtrg_metrics = pd.DataFrame.from_dict(dtrg_metrics, orient='index')
dtrg_metrics.columns = ['Decision Tree Regression']



# Load from file
with open(pkl_filename_rf, 'rb') as file:
    pickle_model_rf = pickle.load(file)


y_test_predict = pickle_model_rf.predict(X_test)
y_train_predict = pickle_model_rf.predict(X_train)

print('R^2 training: %.3f, R^2 test: %.3f' % (
      (metrics.r2_score(y_train, y_train_predict)), 
      (metrics.r2_score(y_test, y_test_predict))))


rf_metrics = {
                    'Root Mean Squared Error' : metrics.mean_squared_error(y_test, y_test_predict)**0.5,
                    'Mean Absolute Error' : metrics.mean_absolute_error(y_test, y_test_predict),
                    'R^2' : metrics.r2_score(y_test, y_test_predict),
                    'Explained Variance' : metrics.explained_variance_score(y_test, y_test_predict)
                }

#return reg_metrics
rf_metrics = pd.DataFrame.from_dict(rf_metrics, orient='index')
rf_metrics.columns = ['Random Forest Regression']




results=pd.concat([polyreg_metrics, lasso_metrics ,df_regr_metrics,dtrg_metrics,rf_metrics], axis=1)
print(results)


result2 = results.to_json('export.json', orient='index')