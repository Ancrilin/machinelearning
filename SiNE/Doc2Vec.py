from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import numpy
import csv

filepath = "data/20ClassesRawData_API_cleanTag.csv"
stop_path = 'data/stop.txt'

name = []
label_dict = {}
label = []
title_dict = {}
doc = []
stop = []

with open(stop_path, 'r', encoding='utf-8')as stp:#停用词表
    for row in stp:
        stop.append(row.strip())

with open(filepath, 'r', encoding="utf-8") as fp:
    reader = csv.reader(fp)
    i = -1
    for row in reader:
        description = row[3]
        t_name = row[2]
        t_name = t_name.replace(" API",'')#除去后面API字样
        name.append(t_name)
        t_label = row[5]
        label.append(t_label)
        description = description.replace('\n','').replace('.','').replace(',','').replace('(','').replace(')','').lower().strip().split(' ')
        for each in description:
            if each in stop:
                description.remove(each)#去除停用词
                # print(each)
        # print(description)
        doc.append(description)
        # print(description)
        if t_name not in title_dict:
            title_dict[t_name] = i
        if t_label not in label_dict:
            label_dict[t_label] = i
        i = i + 1
        # print(row)
    del title_dict['name']
    del label_dict['category']
    name = name[1:]
    label = label[1:]
    doc = doc[1:]
    j = 0
    # print(title_dict)
    # print(label_dict)
    # print(doc[0])

data = []
# print(name)
print(label)
print(label_dict)
for i in range(len(doc)):
    # data.append(TaggedDocument(doc[i], tags=[i]))
    t = []
    t.append(label_dict[label[i]])
    tag = TaggedDocument(doc[i], tags=t)
    data.append(tag)
    print(tag)
    # print(t)
    # print(label_dict[label[i]])
    # print(doc[i])
# print(data)


# model = Doc2Vec(dm=1, min_count=1, window=3, vector_size=200, sample=1e-3, negative=5)
# model.train(data, total_examples=model.corpus_count, epochs=500)
def train(x_train, size=200, epoch_num=1):  ##size 最终训练出的句子向量的维度
    model_dm = Doc2Vec(x_train, min_count=3, window=5, vector_size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=10)
    model_dm.save('model/model_dm')  ##模型保存的位置
    return model_dm

print(len(data))
model  = train(data)

with open("result/description_vec_label.csv", 'w', encoding="utf-8", newline='') as fp:
    header = ['name', 'vector']
    writer = csv.writer(fp)
    writer.writerow(header)
    for each in range(len(data)):
        row = []
        t_name = name[each]
        vec = model.docvecs[each]
        t_label = label[each]
        # print(t_name)
        # print(vec)
        row.append(t_name)
        row.append(vec)
        row.append(t_label)
        writer.writerow(row)

model.wv.save_word2vec_format('result/word_vec.txt',binary = False)
