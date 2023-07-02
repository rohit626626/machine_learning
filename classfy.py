from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
#checking description for data
# print(iris.DESCR)
features=iris.data
labels=iris.target

#checking first feature and lebel
# print(features[0], labels[0])

clf=KNeighborsClassifier()
clf.fit(features, labels)

preds=clf.predict([[3.2,2.3,3.2,1.6]])
print(preds)