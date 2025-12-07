import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('sample_sales.csv', parse_dates=['date'])
df['month'] = df['date'].dt.month + (df['date'].dt.year - df['date'].dt.year.min())*12
X = df[['month']]
y = df['sales']
model = LinearRegression().fit(X,y)
future = pd.DataFrame({'month':[X['month'].max()+i for i in range(1,7)]})
pred = model.predict(future)
print('Future sales predictions:', pred)