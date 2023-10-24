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
from preprocess import preprocess_df, fillna
from model import fit_model
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(df, model, cols):
    model, val = fit_model(df, model, cols)
    preds=model.predict(val.drop('Y', axis=1).values)
    mse=mean_squared_error(val['Y'].values, preds)
    r2=r2_score(val['Y'].values, preds)
    print(mean_squared_error(val['Y'].values, preds))
    print(r2_score(val['Y'].values, preds))
    return model, r2

def generate_preds(test, model, df, cols, final_df, test_cols):
    model, r2=evaluate_model(df, model, cols)
    test=test[test_cols]
    test=fillna(test)
    if r2>=0.5:
        preds=model.predict(test.values)
        test['preds']=preds
        test.to_csv(final_df)
    else:
        print('R2 too low')
