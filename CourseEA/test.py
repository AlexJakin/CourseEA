"""
@author: Alex
@contact: 1272296763@qq.com or jakinmili@gmail.com
@file: test.py
@time: 2019/10/18 16:25
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
with open("embeddings\sgns.zhihu.bigram", 'wb') as new_file, open("embeddings\sgns.zhihu.bigram.bz2",'rb') as file:
    decompressor = bz2.BZ2Decompressor()
    for data in iter(lambda: file.read(100 * 1024), b''):
        new_file.write(decompressor.decompress(data))
