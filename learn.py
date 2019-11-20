import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from featexp import get_univariate_plots
import csv
import pandas

def get_data(filepath):
    with open(filepath, 'r', encoding='utf-8')as fp:
        reader = csv.reader(fp)
        source = []
        for row in reader:
            source.append(row)
        attribute = source[0]
        print(attribute)
        source = source[1:]
        print(source[0])
    return source

filepath = "data/DBP/wiki_data.csv"
get_data(filepath)
