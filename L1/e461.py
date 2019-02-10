import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = pd.read_csv('Smarket.csv', usecols=range(0,10), index_col=0, parse_dates=True)
print(df.describe())
print(df.head())
print(df.corr())

# Plot
plt.plot(df["Year"], df["Volume"], 'rx')
plt.xlabel('Years')
plt.ylabel('Volume')
plt.show()

