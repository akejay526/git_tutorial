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
from model import fit_model
from evaluate import generate_preds
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    test=pd.read_csv('test.csv')
    df=pd.read_csv('train.csv')
    cols=['YrSold','SalePrice', 'GrLivArea', 'BsmtFinSF2', 'Fireplaces', 'YearBuilt', '1stFlrSF', 'PoolArea', 'TotalBsmtSF', 'GarageQual', 'BldgType', 'Exterior2nd', 'BsmtFinType2', 'GarageArea', 'TotRmsAbvGrd', 'MSSubClass', 'GarageCond', 'GarageYrBlt', 'OverallQual', 'GarageCars', 'Exterior1st']
    test_cols=['YrSold','GrLivArea', 'BsmtFinSF2', 'Fireplaces', 'YearBuilt', '1stFlrSF', 'PoolArea', 'TotalBsmtSF', 'GarageQual', 'BldgType', 'Exterior2nd', 'BsmtFinType2', 'GarageArea', 'TotRmsAbvGrd', 'MSSubClass', 'GarageCond', 'GarageYrBlt', 'OverallQual', 'GarageCars', 'Exterior1st']
    model = LinearRegression()

    generate_preds(test, model, df, cols, 'housing_test.csv', test_cols)
