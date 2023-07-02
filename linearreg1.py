import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

ln_x=np.array([[1], [2], [3]])

ln_x_train=ln_x
ln_x_test=ln_x
ln_y_train=np.array((3,2,4))
ln_y_test=np.array((3,2,4))

model=linear_model.LinearRegression()
model.fit(ln_x_train, ln_y_train)

ln_y_predicted=model.predict(ln_x_test)
print("Mean squared error is: ", mean_squared_error(ln_y_test,ln_y_predicted))

print("Weight: ", model.coef_)
print("Intercept: ", model.intercept_)

plt.scatter(ln_x_test, ln_y_test)
plt.plot(ln_x_test, ln_y_predicted)
plt.show()


