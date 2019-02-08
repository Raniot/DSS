from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm # import statsmodels 

boston = load_boston()

bos = pd.DataFrame(boston["data"])
bos.columns = boston["feature_names"]
bos['MDEV'] = boston["target"]
print(bos.head())

lm = LinearRegression()
X = bos['LSTAT']
Y = bos['MDEV']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

model = sm.OLS(Y,X).fit()



# model = lm.fit(X, Y)
print(model.summary())
print(X.shape)
print(Y.shape)
