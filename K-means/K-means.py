import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import random

iris = datasets.load_iris()
X = iris.data[:, 2:4]

# print(X.shape)
# # plt.scatter(X[:, 0], X[:, 1], c = "red", marker='o', label='see')
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.legend(loc=2)
# plt.show()

class Kmeans():
    def __init__(self, n_cluster, tol=1e-4, n_init=10, step=150):
        self.k = n_cluster
        self.tol = tol
        self.n_init = n_init  # 进行多次聚类，选择最好的一次
        self.times = 0
        self.step = step

    def fit(self, dataset):
        centroid = self.init_centers(dataset)
        print("init centers")
        print(centroid)
        labels = self.assign_points(dataset, centroid)
        self.show(labels, dataset, centroid)
        # t_centers = self.update_centers(labels, dataset)
        self.train(labels, dataset, centroid)
        # self.update_centers(labels, dataset, centers)

    def train(self, labels, dataset, centroid):
        for i in range(self.step):
            print(i)
            t_centroid = self.update_centers(labels, dataset, centroid)
            print("centroid")
            print(centroid)
            print("t_centroid")
            print(t_centroid)
            if self.isStable(centroid, t_centroid):
                return labels, centroid
            centroid = t_centroid
            self.show(labels, dataset, centroid)
            labels = self.assign_points(dataset, centroid)

    def isStable(self, centroid_1, centroid_2):
        # print("centroid_1")
        # print(centroid_1)
        # print("centroid_2")
        # print(centroid_2)
        centroid_1 = np.array(centroid_1)
        centroid_2 = np.array(centroid_2)
        result = np.abs(centroid_1 - centroid_2)
        # print("result")
        # print(result)
        return np.all(result < self.tol)

    def getDistance(self, vecA, vecB):#计算欧氏距离
        vecA = np.array(vecA)
        vecB = np.array(vecB)
        result = np.sum(np.square(vecA - vecB))
        return result

    def init_centers(self, dataset):#初始化均值点，随机选择
        num = []
        centroid = []
        for i in range(self.k):
            flag = 1
            while (flag):
                t = random.randint(0, len(dataset))
                if t not in num:
                    num.append(t)
                    flag = 0
                    centroid.append(dataset[t])
        return centroid

    def assign_points(self, dataset, centroid):
        labels = []
        for i in range(len(dataset)):
            shortest = float("inf")  # 正无穷
            shortest_index = -1
            for j in range(self.k):
                t_distance = self.getDistance(dataset[i], centroid[j])
                if t_distance < shortest:#选择最近距离的均值点
                    shortest = t_distance
                    shortest_index = j
            labels.append(shortest_index)
        return labels

    def update_centers(self, labels, dataset, centroid):#更新均值点
        t_centroid = []
        labels = np.array(labels)
        dataset = np.array(dataset)
        for i in range(self.k):
            avg = np.mean(dataset[labels == i], axis=0)
            # print(avg)
            if np.all(avg != float("nan")):#防止出现labels中没有分到self.k个类
                t_centroid.append(avg)
            else:
                t_centroid.append(centroid[i])
        return t_centroid

    def show(self, labels, dataset, centroid):
        labels = np.array(labels)
        dataset = np.array(dataset)
        x0 = dataset[labels == 0]
        x1 = dataset[labels == 1]
        x2 = dataset[labels == 2]
        # print(x0)
        # print(x2)
        plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
        plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
        plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
        plt.scatter(centroid[0][0], centroid[0][1], c="black", label='point0')
        plt.scatter(centroid[1][0], centroid[1][1], c="yellow", label='point1')
        plt.scatter(centroid[2][0], centroid[2][1], c="pink", label='point2')
        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.legend(loc=2)
        plt.title("graph: " + str(self.times))
        plt.show()
        self.times = self.times + 1


# estimator = KMeans(n_clusters=3)#构造聚类器
# estimator.fit(X)#聚类
# label_pred = estimator.labels_ #获取聚类标签
# #绘制k-means结果
# x0 = X[label_pred == 0]
# x1 = X[label_pred == 1]
# x2 = X[label_pred == 2]
# plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
# plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
# plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.legend(loc=2)
# plt.show()

# print(label_pred)
a = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
b = [0, 0, 0, 1, 1, 1, 2, 2, 2]
a = np.array(a)
b = np.array(b)
# print(a[b == 0])
# print(np.mean(a[b==1], axis=0))

# a = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]])
# centers = [[1,2],[3,4]]
estimator = Kmeans(3)
# estimator.update_centers(b, a, centers)
estimator.fit(X)
# print(len(X))
# x=np.array([1,2,3])
# y=np.array([4,5,6])
# print(estimator.isStable(x, y))
# print(np.abs(x-y))
# print(np.sum(np.square(x-y)))
