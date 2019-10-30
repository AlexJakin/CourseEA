"""
@author: Alex
@contact: 1272296763@qq.com or jakinmili@gmail.com
@file: DataPreProcess.py
@time: 2019/10/18 15:28
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba
from gensim.models import KeyedVectors
import warnings
import bz2
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

class DataProcess:

    def __init__(self, dataset, embedding_dir):
        self.dataset = dataset
        self.embedding_dim = 300 # 根据预训练词向量
        self.embedding_matrix = None
        self.embedding_matrix_flag = False
        self.__embedding_dir = embedding_dir
        self.__embedding_dir_new = embedding_dir+".bz2"
        self.cn_model = self.decompressor()


    def decompressor(self):
        print("预计需要几分钟的加载时间。。。")
        with open(self.__embedding_dir, 'wb') as new_file, open(self.__embedding_dir_new,'rb') as file:
            decompressor = bz2.BZ2Decompressor()
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(decompressor.decompress(data))

        #使用gensim加载预训练中文分词embedding
        cn_model = KeyedVectors.load_word2vec_format('embeddings/sgns.zhihu.bigram',
                                                     binary=False, unicode_errors="ignore")
        print("加载完成")
        return cn_model

    def get_cn_model(self):
        return self.cn_model

    def process(self):
        assert len(self.dataset) != 0,  "数据集为空"

        train_texts_origin = []
        train_score = []
        train_target = []
        """
        dataset: [username, comment, comment, label] 0 消极 1 积极
        """
        for row in self.dataset:
            train_texts_origin.append(row[1])
            train_score.append(float(row[2]))
            train_target.append(int(row[3]))

        return train_texts_origin, train_score, train_target

    def dataset_to_train_token(self, train_texts_origin, train_score, train_target):
        """
        将数据集转化为词典索引列表
        :param train_texts_origin:
        :param train_score:
        :param train_target:
        :return:
        """
        train_tokens = []
        for i, text in enumerate(train_texts_origin):
            # 去掉标点
            text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
            text = re.sub('[^\w\u4e00-\u9fff]+', '', text)
            text = re.sub('[0-9]', '', text)
            # 结巴分词
            cut = jieba.cut(text)
            # 结巴分词的输出结果为一个生成器
            # 把生成器转换为list
            cut_list = [i for i in cut]
            # 去除例如6666这样子的评论
            if len(cut_list) == 1:
                del train_score[i]
                del train_target[i]
            elif len(cut_list) == 0:
                del train_score[i]
                del train_target[i]
            else:
                for i, word in enumerate(cut_list):
                    try:
                        # 将词转换为索引index
                        cut_list[i] = self.cn_model.vocab[word].index
                    except KeyError:
                        # 如果词不在字典中，则输出0
                        cut_list[i] = 0
                train_tokens.append(cut_list)
        return train_tokens, train_score, train_target

    def get_tokens_num_and_distribution(self, train_tokens):
        # 获得所有tokens的长度
        num_tokens = [len(tokens) for tokens in train_tokens]
        num_tokens = np.array(num_tokens)
        plt.hist(np.log(num_tokens), bins=100)
        plt.xlim((0, 10))
        plt.ylabel('number of tokens')
        plt.xlabel('length of tokens')
        plt.title('Distribution of tokens length')
        plt.show()
        return num_tokens

    def tokens_process(self, train_tokens, num_tokens):
        # 取tokens平均值并加上两个tokens的标准差，
        # 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖99%左右的样本
        max_tokens = np.mean(num_tokens) + 3 * np.std(num_tokens)
        max_tokens = int(max_tokens)
        print("使用正太分布3δ法则可覆盖", np.sum( num_tokens < max_tokens ) / len(num_tokens))

        # 只使用前50000个词
        num_words = 50000
        # 初始化embedding_matrix，之后在keras上进行应用
        embedding_matrix = np.zeros((num_words, self.embedding_dim))
        # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
        # 维度为 50000 * 300
        for i in range(num_words):
            embedding_matrix[i, :] = self.cn_model[self.cn_model.index2word[i]]
        self.embedding_matrix = embedding_matrix.astype('float32')
        train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                                  padding='pre', truncating='pre')
        # 超出五万个词向量的词用0代替
        train_pad[train_pad >= num_words] = 0
        return train_pad

    def get_embedding_matrix(self):
        assert self.embedding_matrix_flag == True, "需先执行tokens_process()函数"
        return self.embedding_matrix

    def reverse_tokens(self, tokens):
        """
        用来将tokens转换为文本
        :param tokens:
        :return:
        """
        text = ''
        for i in tokens:
            if i != 0:
                text = text + (self.cn_model.index2word[i])
            else:
                text = text + ' '
        return text

