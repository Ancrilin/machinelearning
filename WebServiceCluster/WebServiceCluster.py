import numpy as np
import csv
from sklearn.cluster import KMeans

RawData_path = "data/20ClassesRawData_API_cleanTag.csv"
doc2Vec_path = "data/description_vec_csv_stop.csv"
node2Vec_path = "data/node_emb_csv.csv"

name2id = {}
label2id = {}
name = []
label = []
name_label_dic = {}#名称对应于类别的字典
data_dic = {}

def get_label():
    source = []
    with open(RawData_path, 'r', encoding='utf-8')as fp:
        reader = csv.reader(fp)
        for row in reader:
            source.append(row)
    source = source[1:]
    i = 0
    for row in source:
        category = row[5].strip()
        name = row[2].replace(' API','').strip()
        if name not in name_label_dic:
            name_label_dic[name] = category
        if category not in label2id:
            label2id[category] = i
            i = i + 1
        label.append(category)
    print("len(label)", len(label))
    # print(label)


def get_info():#获取名称和对应的字典
    source = []
    with open(doc2Vec_path, 'r', encoding='utf-8')as fp:
        reader = csv.reader(fp)
        for row in reader:
            source.append(row)
        source = source[1:]
        j = 0
        for row in source:
            row[0] = row[0].strip()
            name.append(row[0])
            if row[0] not in name2id:
                name2id[row[0]] = j
                j = j + 1

def loadVec(filepath):
    data = []
    dic = {}
    source = []
    with open(filepath, 'r', encoding='utf-8')as fp:
        reader = csv.reader(fp)
        for row in reader:
            source.append(row)
        source = source[1:]
    j = 0
    for row in source:
        row[1] = row[1].replace('\n', '').replace('/', '').replace('[','').replace(']','')
        row[1] = row[1].split(' ')
        row[0] = row[0].replace(' API','').strip()
        vec = []
        for i in range(len(row[1])):
            if row[1][i] != '':
                vec.append(float(row[1][i]))
        dic[row[0]]=vec
        # print(type(vec[0]))
    del source
    # print(dic)
    return dic

def concat(node_dic, doc_dic):#连接
    data = []
    for each in name:
        vec = []
        doc = doc_dic[each]
        node = node_dic[each]
        vec = doc + node
        data.append(vec)
        # vec = np.array(vec)
        data_dic[each]=vec
    data = np.array(data)
    return data

def getClusterLabel(result2name_cluster):#获得簇中最大的类别的数量和对应的类别
    dic = {}
    # print("len cluster", len(result2name_cluster))
    for each in result2name_cluster:
        if each not in dic:
            dic[name_label_dic[each]] = 0
    for each in result2name_cluster:
        dic[name_label_dic[each]] = dic[name_label_dic[each]] + 1
    print("dic", dic)
    max = dic[name_label_dic[result2name_cluster[0]]]
    max_label = list(dic.keys())[0]
    # print(max, max_label)
    # for each in dic:
    #     if dic[each] > max:
    #         max = dic[each]
    for i in range(len(list(dic.keys()))):
        if dic[list(dic.keys())[i]] > max:
            max = dic[list(dic.keys())[i]]
            max_label = list(dic.keys())[i]
    print("max", max, "max_label", max_label)
    return max,max_label

def accuracy(result2name):
    p = 0
    i = 0
    for cluster in result2name:
        print("Cluster ", i)
        i = i + 1
        s = 0
        n = len(cluster)
        num, label = getClusterLabel(cluster)
        for each in cluster:
            if name_label_dic[each] == label:
                s = s + 1
        pk = s / n
        p = p + pk
        print("pk", pk)
    print("Final")
    # print(p)
    p = p / len(result2name)
    print("p", p)
    return p

get_info()
get_label()
doc_dic = loadVec(doc2Vec_path)
node_dic = loadVec(node2Vec_path)
data = concat(node_dic, doc_dic)
print("len(data)",len(data))
model = KMeans(n_clusters=20, init="k-means++", n_init=10, max_iter=500, tol=0.001)
model.fit(data)
label_pred = model.labels_
print("len(label_pred) ", len(label_pred))
print(label_pred)#已聚类后的数据标签
result = []#创建存储20个类的列表
result2name = []#转换为名称
for i in range(20):
    t = []
    result.append(t)
for i in range(len(label_pred)):
    result[label_pred[i]].append(i)
for cluster in result:
    t = []
    for each in cluster:
        t.append(name[each])
    result2name.append(t)
# t = []
# for i in range(len(label_pred)):
#     if label_pred[i] == 0:
#         t.append(i)
# print(name)
# print(result)
# print(name_label_dic)
# print(result2name)
accuracy(result2name)