import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
#Import our own function
from preprocess import preprocess_df
from sklearn.model_selection import train_test_split

def fit_model(df, model, cols):
    df=preprocess_df(df, cols)
    train,val=train_test_split(df, train_size=0.8)
    X = train.drop('Y', axis=1).values
    y = train['Y'].values
    model.fit(X, y)
    return model, val

def evaluate_model(df, model, cols):
    model, val = fit_model(df, model, cols)
    preds=model.predict(val.drop('Y', axis=1).values)
    mse=mean_squared_error(val['Y'].values, preds)
    r2=r2_score(val['Y'].values, preds)
    print(mean_squared_error(val['Y'].values, preds))
    print(r2_score(val['Y'].values, preds))
    return model, r2
