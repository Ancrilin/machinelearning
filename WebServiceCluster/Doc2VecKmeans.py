import numpy as np
import csv
from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

filepath = "data/description_vec_label.csv"
classrawdata = "data/20ClassesRawData_API_cleanTag.csv"
stop_path = 'data/stop.txt'

doc_TaggedDocuments= []
name = []
name_label_dic = {}#名称对应于类别的字典
label = []
label_dict = {}
name_tags_dict = {}
doc = []
vec = []

# with open(filepath, 'r', encoding='utf-8')as fp:
#     reader = csv.reader(fp)
#     source = []
#     for row in reader:
#         source.append(row)
#     source = source[1:]
#     for each in source:
#         if each[0] not in name_label_dic:
#             name_label_dic[each[0].strip()] = each[2]
#         vec = []
#         each[1] = each[1].replace('\n', '').replace('/', '').replace('[', '').replace(']', '')
#         each[1] = each[1].split(' ')
#         # print(each[1])
#         for i in each[1]:
#             if i != '':
#                 vec.append(float(i))
#         data.append(vec)
#         name.append(each[0].strip())

def getdata(filepath):
    stop = []
    with open(stop_path, 'r', encoding='utf-8')as stp:  # 停用词表
        for row in stp:
            stop.append(row.strip())
    with open(filepath, 'r', encoding='utf-8')as fp:
        reader = csv.reader(fp)
        source = []
        for row in reader:
            source.append(row)
        source = source[1:]
        i = 0
        for row in source:
            t_name = row[2].replace('API','').strip()#获得名称，按顺序
            name.append(t_name)
            t_label = row[5].strip()#获得类别标签，按顺序
            label.append(t_label)
            if t_label not in label_dict:#对每个类别编号
                label_dict[t_label] = i
            i = i + 1
            if t_name not in name_label_dic:
                name_label_dic[t_name] = t_label#名称对应于类别的字典
            t_tags = row[4].strip().split(',')#获得tags
            name_tags_dict[t_name]=t_tags
            description = row[3]#获得描述,按顺序
            description = description.replace('\n','').replace('.','').replace(',','').replace('(','').replace(')','').lower().strip().split(' ')#分词
            for each in description:
                if each in stop:
                    description.remove(each)  # 去除停用词
            doc.append(description)#分词后


def getTaggedDocuments():
    for i in range(len(doc)):
        t = []
        t.append(label_dict[label[i]])
        # tag = TaggedDocument(doc[i], tags=t)
        tag = TaggedDocument(doc[i], tags=[i])
        doc_TaggedDocuments.append(tag)
        # print(tag)


def getVec(x_train, size=500, epoch_num=1):#doc2vec训练获得文本向量
    model_dm = Doc2Vec(x_train, min_count=3, window=5, vector_size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=1)
    model_dm.save('model/model_dm')  ##模型保存的位置
    for i in range(len(doc_TaggedDocuments)):
        t_vec = model_dm.docvecs[i]
        vec.append(t_vec)
    # print(vec)
    model_dm.wv.save_word2vec_format('data/word_vec.txt', binary=False)
    return model_dm

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

#获得doc2vec文本向量
getdata(classrawdata)#基本数据
getTaggedDocuments()#获得TaggedDocument
model_vec = getVec(doc_TaggedDocuments)#训练获得文本向量
print(name)
print(name_label_dic)
model = KMeans(n_clusters=20, init="k-means++", n_init=20, max_iter=500, tol=0.001)
data = np.array(vec)
model.fit(vec)
#进行kmean++聚类
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
accuracy(result2name)
