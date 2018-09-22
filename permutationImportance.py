import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import eli5


reg = linear_model.LinearRegression()

ar = np.array([[[1],[2],[3]], [[2],[4],[6]]])

y = ar[1,:]

x = ar[0,:]

model=reg.fit(x,y)

print('Coefficients: \n', reg.coef_)

xTest = np.array([[4],[5],[6]])
ytest =  np.array([[8],[10],[12]])

preds = reg.predict(xTest)

print('Coefficients: \n', reg.coef_)

print("Mean squared error: %.2f" % mean_squared_error(ytest,preds))

eli5.show_weights(model)
