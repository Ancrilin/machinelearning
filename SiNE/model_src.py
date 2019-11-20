import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import torch.optim as optim


def hadamard(x, y):
    return x * y


def average(x, y):
    return (x + y)/2.0


def l1(x, y):
    return np.abs(x - y)


def l2(x, y):
    return np.power(x - y, 2)


def concat(x, y):#数组拼接
    return np.concatenate((x, y), axis=1)


FEATURE_FUNCS = {
    'l1': l1,
    'l2': l2,
    'concat': concat,
    'average': average,
    'hadamard': hadamard
}


class SiNE(nn.Module):
    def __init__(self, num_nodes, dim1, dim2):
        super(SiNE, self).__init__()
        # self.cuda()#model_cuda
        self.tanh = nn.Tanh()
        # num_embeddings(int) - 嵌入字典的大小
        # embedding_dim(int) - 每个嵌入向量的大小
        self.embeddings = nn.Embedding(num_nodes + 1, dim1)#一个保存了固定字典和大小的简单查找表。
        # self.embeddings = self.embeddings.cuda()#embedding_cuda
        self.layer11 = nn.Linear(dim1, dim2, bias=False)#对输入数据做线性变换：y=Ax+b 第一层隐藏层
        # self.layerl1 = self.layer11.cuda()#layerl1_cuda
        self.layer12 = nn.Linear(dim1, dim2, bias=False)
        # self.layerl2 = self.layer12.cuda()# layerl2_cuda
        self.bias1 = Parameter(torch.zeros(1))
        # self.bias1 = self.bias1.cuda()#bias1_cuda
        self.layer2 = nn.Linear(dim2, 1, bias=False)#第二层隐藏层
        # self.layer2 = self.layer2.cuda()#layer2_cuda
        self.bias2 = Parameter(torch.zeros(1))#参数
        # self.bias2 = self.bias2.cuda()#bias2_cuda
        self.register_parameter('bias1', self.bias1)#向module添加 parameter
        self.register_parameter('bias2', self.bias2)

    def forward(self, xi, xj, xk, delta):#前向传播
        i_emb = self.embeddings(xi)
        j_emb = self.embeddings(xj)
        k_emb = self.embeddings(xk)

        z11 = self.tanh(self.layer11(i_emb) + self.layer12(j_emb) + self.bias1)
        z12 = self.tanh(self.layer11(i_emb) + self.layer12(k_emb) + self.bias1)

        f_pos = self.tanh(self.layer2(z11) + self.bias2)
        f_neg = self.tanh(self.layer2(z12) + self.bias2)

        zeros = Variable(torch.zeros(1))

        loss = torch.max(zeros, f_pos + delta - f_neg)
        loss = torch.sum(loss)
        # loss.cuda()

        return loss#总误差

    def _regularizer(self, x):#返回范数
        zeros = torch.zeros_like(x)
        normed = torch.norm(x - zeros, p=2)#返回输入张量input 的p 范数。
        term = torch.pow(normed, 2)
        # print('The parameter of ', x)
        # print('Yields ',term)
        return term

    def regularize_weights(self):
        loss = 0
        for parameter in self.parameters():
            loss += self._regularizer(parameter)
        return loss

    def get_embedding(self, x):
        x = Variable(torch.LongTensor([x]))
        emb = self.embeddings(x)
        emb = emb.data.numpy()[0]
        return emb

    def get_edge_feature(self, x, y, operation='hadamard'):
        func = FEATURE_FUNCS[operation]
        x = self.get_embedding(x)
        y = self.get_embedding(y)
        return func(x, y)




def tensorfy_col(x, col_idx):
    col = x[:,col_idx]
    col = torch.LongTensor(col)
    col = Variable(col)
    return col


def get_training_batch(triples, batch_size):
    nrows = triples.shape[0]
    rows = np.random.choice(nrows, batch_size, replace=False)#从nrows中随机抽取batch_size个，不放回，即不重复
    choosen = triples[rows,:]#去rows中对应的所有数据
    xi = tensorfy_col(choosen, 0)#选取index0
    xj = tensorfy_col(choosen, 1)
    xk = tensorfy_col(choosen, 2)
    return xi, xj, xk


def fit_model(sine, triplets, delta, batch_size, epochs, alpha,
                lr=0.4, weight_decay=0.0, print_loss=True):
    optimizer = optim.Adagrad(sine.parameters(), lr=lr, weight_decay=weight_decay)#实现Adagrad算法。
    for epoch in range(epochs):
        sine.zero_grad()#将module中的所有模型参数的梯度设置为0
        xi, xj, xk = get_training_batch(triplets, batch_size)
        loss = sine(xi, xj, xk, delta)
        # print(loss)
        regularizer_loss = alpha * sine.regularize_weights()#alpha为控制正则化参数
        # print(regularizer_loss)
        loss += regularizer_loss
        loss.backward()
        optimizer.step()#更新所有的参数，进行单次优化
        if print_loss:
            print('Loss at epoch ', epoch + 1, ' is ', loss.data.item())
    return sine



