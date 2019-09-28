import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class KmeansPPlus():
    def __init__(self, n_cluster = 3, tol=1e-5, step=150):
        self.k = n_cluster              #类别数量
        self.step = step                #最大步长
        self.tol = tol                  #质心偏移精度
        self.times = 0                  #图片编号

    def fit(self, dataset):
        self.times = 0
        dataset = np.array(dataset)
        centroids = self.init_centroids(dataset)
        print("init centroids")
        print(centroids)
        labels = self.assign_points(dataset, centroids)
        print("init labels")
        print(labels)
        self.show(labels, dataset, centroids)
        self.train(dataset, labels, centroids)

    def train(self, dataset, labels, centroids):
        print("training")
        for i in range(self.step):
            print("training step: " + str(i))
            t_centroids = self.update_centroids(dataset, labels, centroids)
            if self.isStable(centroids, t_centroids):   #判断质心偏移是否超过规定精度
                return labels, centroids
            centroids = t_centroids
            self.show(labels, dataset, centroids)
            labels = self.assign_points(dataset, centroids)#继续分配点到最近的质心
            self.times = self.times + 1

    def isStable(self, centroid_1, centroid_2):
        result = np.abs(centroid_1 - centroid_2)
        return np.all(result < self.tol)

    def get_distance(self, vecA, vecB):
        return np.square(np.sum(np.square(vecA - vecB)))

    def assign_points(self, dataset, centroids):#选择距离最近的质点
        labels = np.zeros(len(dataset), dtype=np.int)
        for i in range(len(dataset)):
            shortest = np.inf#初始化最近距离为正无穷
            shortest_index = -1
            for j in range(len(centroids)):
                t_shortest = self.get_distance(dataset[i], centroids[j])
                if t_shortest < shortest:
                    shortest_index = j
                    shortest = t_shortest
            labels[i] = shortest_index
        return labels

    def update_centroids(self, dataset, labels, centroids):#更新质心
        t_centroids = np.array(centroids)
        print("labels")
        print(labels)
        for i in range(len(centroids)):
            centroid_mean = np.mean(dataset[labels == i], axis=0)
            if np.all(centroid_mean != np.nan):#防止出现没有任何点分配到某个质心
                t_centroids[i] = centroid_mean
        return t_centroids

    def init_centroids(self, dataset):#初始化质心，出了第一个随机，其余根据概率选择
        rand = np.random.randint(0, len(dataset))
        centroids = np.zeros([self.k, len(dataset[0])], dtype=np.float)
        centroids[0] = dataset[rand]
        for i in range(1, self.k):
            centroids[i] = self.get_centroid(dataset, centroids, i)#获取第k个初始化的质心
        return centroids

    def probability(self, points_distance):#根据点到最近质心的距离来获取概率前缀和
        points_probability = np.zeros(len(points_distance) + 1, dtype=np.float)#初始化所有点的概率都为0
        distance_sum = np.sum(points_distance)#距离和，频率概率的分母
        for i in range(1, len(points_probability)):
            points_probability[i] = points_distance[i-1] / distance_sum + points_probability[i-1]
        return points_probability[1:]

    def get_centroid(self, dataset, centroids, x):#得到第k个初始化的质心
        points_distance = np.zeros(len(dataset), dtype=np.float)
        for i in range(len(dataset)):
            shortest = np.inf#初始化最大距离为正无穷
            for j in range(x):#得到点与最近质心的距离
                t_shortest = self.get_distance(dataset[i], centroids[j])
                if t_shortest < shortest:
                    shortest = t_shortest
            points_distance[i] = shortest
        points_probablity = self.probability(points_distance)#得到概率前缀和
        rand = np.random.random_sample()#0-1随机数
        centroid = np.zeros(2, dtype=np.float)
        for i in range(len(dataset)):#轮转法选择初始化的质心
            if rand <= points_probablity[i]:
                centroid = dataset[i]
                break#找到区间后跳出循环
        return centroid

    def show(self, labels, dataset, centroids):#显示图片
        x0 = dataset[labels == 0]
        x1 = dataset[labels == 1]
        x2 = dataset[labels == 2]
        plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
        plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
        plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
        plt.scatter(centroids[0][0], centroids[0][1], c="black", label='point0')
        plt.scatter(centroids[1][0], centroids[1][1], c="yellow", label='point1')
        plt.scatter(centroids[2][0], centroids[2][1], c="pink", label='point2')
        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.legend(loc=2)
        plt.title("graph: " + str(self.times))
        plt.show()
        self.times = self.times + 1

iris = datasets.load_iris()
X = iris.data[:, 2:4]
estimator = KmeansPPlus(n_cluster=3)
estimator.fit(X)