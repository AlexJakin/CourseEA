"""
@author: Alex
@contact: 1272296763@qq.com or jakinmili@gmail.com
@file: CPreprocess.py
@time: 2019/10/12 0:16
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

class File:

    def __init__(self, files_root_dir):
        self.__files_root_dir = files_root_dir
        self.__files_dir_list, self.__files_name_list\
            = self.get_files()


    def get_files(self):
        files_dir_list = []
        files_name_list = []
        for root, dirs, files in os.walk(self.__files_root_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.csv':
                    files_dir_list.append(self.__files_root_dir + file)
                    files_name_list.append(file)
        return files_dir_list, files_name_list

    def get_files_dir_list(self):
        return self.__files_dir_list

    def get_files_name_list(self):
        return self.__files_name_list


class Data:

    def __init__(self, files_dir_list):
        self.__files_dir_list = files_dir_list
        self.data_list = []
        self.__dataset_shape = None
        self.__pos_num = 0
        self.__neg_num = 0

    def get_dataset(self):
        """
        :return: [sample, username, comment, star, pos | neg]
        """
        for file_path in self.__files_dir_list:
            csv_reader = csv.reader(open(file_path, encoding='utf-8'))
            for row in csv_reader:
                self.data_list.append(row)
        return self.data_list

    def get_dataset_shape(self):
        assert len(self.data_list) != 0 , \
        "数据集为空，或者需要先执行get_dataset()函数，才可以使用get_dataset_shape()获取数据集的形状"
        self.__dataset_shape = np.array(self.data_list).shape
        return self.__dataset_shape

    def get_dataset_distribution(self):
        assert len(self.data_list) != 0, \
            "数据集为空，或者需要先执行get_dataset()函数，才可以使用get_dataset_shape()获取数据集的形状"
        pos_num = 0
        neg_num = 0
        for row in self.data_list:
            if row[3] == '1':
                pos_num += 1
            elif row[3] == '0':
                neg_num += 1
        print("积极评论数：{}；消极评论数：{}".format(pos_num, neg_num))

        # 柱形图
        plt.rcParams['font.sans-serif'] = ['SimHei']
        X = ["积极", "消极"]
        Y = [pos_num, neg_num]
        plt.bar(X, Y, 0.4)
        plt.xlabel("评论情绪")
        plt.ylabel("数量")
        plt.title("情绪分布")

        plt.show()


