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


def fillna(df):
    n_df=df.select_dtypes(include='number')
    n_cols=n_df.columns
    for col in n_cols:
        df[col]=df[col].fillna(df[col].median())
    s_df=df.select_dtypes(include='object')
    s_cols=s_df.columns
    for col in s_cols:
        df[col]=df[col].fillna(df[col].value_counts().idxmax())
    for col in s_cols:
        df[col]=pd.factorize(df[col])[0]  
    return df   

def preprocess_df(df, cols):
    df=df[cols]
    df=fillna(df)
    df = df.rename(columns={"SalePrice":"Y"})
    return df

