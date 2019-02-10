from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

boston = load_boston()

print(boston["DESCR"])

bos = pd.DataFrame(boston["data"])
bos.columns = boston["feature_names"]
bos['MDEV'] = boston["target"]

print(bos.head())

X = bos[['LSTAT', 'AGE']]
Y = bos['MDEV']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

model = sm.OLS(Y,X).fit()
print(model.summary())

# Part 2

X = bos[boston["feature_names"]]
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
model = sm.OLS(Y,X).fit()
print(model.summary())

# Part 3
vifResult = pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

print(vifResult)