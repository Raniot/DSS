from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm # import statsmodels 
import matplotlib.pyplot as plt
import seaborn as sns

boston = load_boston()

print(boston["DESCR"])

bos = pd.DataFrame(boston["data"])
bos.columns = boston["feature_names"]
bos['MDEV'] = boston["target"]
print(bos.head())

lm = LinearRegression()
XInit = bos['LSTAT']
Y = bos['MDEV']
X = sm.add_constant(XInit) ## let's add an intercept (beta_0) to our model

model = sm.OLS(Y,X).fit()



# model = lm.fit(X, Y)
print(model.summary())
print(X.shape)
print(Y.shape)

testData = [5, 10, 15]
testData = sm.add_constant(testData)

result = model.predict(testData)
print(result)

print(XInit.idxmax())

plt.figure(figsize=(12,10))
sns.regplot(XInit, Y,robust=True)
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show()



