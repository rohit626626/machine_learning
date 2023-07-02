from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris=datasets.load_iris()
# print(iris["data"])
# print(iris["target"])
# print(iris["DESCR"])
# print(list(iris.keys()))

x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)

# train model
clf=LogisticRegression()
clf.fit(x,y)
example=clf.predict(([[2.6]]))
print(example)

#matplotlib
x_new=np.linspace(0,3,1000).reshape(-1,1)
y_pro=clf.predict_proba(x_new)
print(y_pro)
plt.plot(x_new, y_pro[:,1])
plt.show()




# print(y)
# print(x)