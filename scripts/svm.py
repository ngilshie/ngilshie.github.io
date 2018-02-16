# Script to create plots for RKHS blogpost.
# 11 Feb 2018

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

np.random.seed(0)
# create first non-linearly-separable dataset
# N = 200 # number of points per class
# X = np.zeros(shape=(N*2, 3))    # 100 points per class and 2 features
# Y = np.repeat([0, 1], repeats=N)   # label points 0 or 1
# radius0 = np.random.normal(loc=1, scale=0.15, size=N)    # sample radius for inner group of points
# theta0 = np.linspace(start=0, stop=2*math.pi, num=N)
# radius1 = np.random.normal(loc=2, scale=0.25, size=N)    # sample radius for outer group of points
# theta1 = np.linspace(start=0, stop=2*math.pi, num=N)
# radius, theta = np.concatenate([radius0, radius1]), np.concatenate([theta0, theta1])
# X[:, 0] = radius * np.cos(theta)   # convert from polar to cartesian coordinates
# X[:, 1] = radius * np.sin(theta)

# plt.figure(figsize=(6,4))
# plt.plot(X[Y==0, 0], X[Y==0, 1], 'bo', markersize=3)
# plt.plot(X[Y==1, 0], X[Y==1, 1], 'ro', markersize=3)
# plt.xlabel('x1'); plt.ylabel('x2')
# plt.show()

# # map to feature space
# X[:, 2] = np.square(X[:, 0]) + np.square(X[:, 1])
# # fit SVM
# clf = svm.SVC(kernel='linear')
# clf.fit(X, Y)
# coef, intercept = clf.coef_[0], clf.intercept_
# print(clf.score(X, Y))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[Y==0, 0], X[Y==0, 1], X[Y==0, 2], 'b')
# ax.scatter(X[Y==1, 0], X[Y==1, 1], X[Y==1, 2], 'r')
# xx, yy = np.meshgrid(range(-3, 4), range(-3, 4))
# z = (-coef[0]*xx - coef[1]*yy - intercept)/coef[2]
# ax.plot_surface(xx, yy, z, color='green', alpha=0.3)
# ax.view_init(10, 30)

# ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('x3')
# plt.show()

# build spiral data
N = 200
X = np.zeros(shape=(N*2, 2))   
Y = np.repeat([0, 1], repeats=N)
radius0 = np.linspace(0, 2, N)
theta0 = np.linspace(0, 3*math.pi, N) + np.linspace(0, 0.7, N)*np.random.randn(N)
radius1 = np.linspace(0, 2, N)
theta1 = np.linspace(math.pi, 4*math.pi, N) + np.linspace(0, 0.7, N)*np.random.randn(N)
radius, theta = np.concatenate([radius0, radius1]), np.concatenate([theta0, theta1])
X[:, 0] = radius * np.cos(theta) 
X[:, 1] = radius * np.sin(theta)

# plt.figure(figsize=(6,4))
# plt.plot(X[Y==0, 0], X[Y==0, 1], 'bo', markersize=3)
# plt.plot(X[Y==1, 0], X[Y==1, 1], 'ro', markersize=3)
# plt.xlabel('x1'); plt.ylabel('x2')
# plt.show()

# fit SVM
clf2 = svm.SVC(kernel='rbf', gamma=1.0, C=5.0)
clf2.fit(X, Y)
print(clf2.score(X, Y))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 0].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xnew = np.c_[xx.ravel(), yy.ravel()]
ynew = clf2.predict(xnew).reshape(xx.shape)

plt.figure(figsize=(6,4))
plt.plot(X[Y==0, 0], X[Y==0, 1], 'bo', markersize=3)
plt.plot(X[Y==1, 0], X[Y==1, 1], 'ro', markersize=3)
plt.xlabel('x1'); plt.ylabel('x2')
CM = plt.cm.get_cmap('Paired')
CM._init()
CM._lut	[:,-1] = 0.2
plt.set_cmap(CM)
plt.pcolormesh(xx, yy, ynew)
plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()