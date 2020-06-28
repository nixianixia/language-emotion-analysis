import jieba
import time
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import jieba
import jieba.analyse
from gensim.models import Word2Vec

with open('./stopwords.txt', encoding='utf8') as file:
    line_list = file.readlines()
    stopword_list = [k.strip() for k in line_list]
    stopword_set = set(stopword_list)
    print('停顿词列表，即变量stopword_list中共有%d个元素' % len(stopword_list))
    print('停顿词集合，即变量stopword_set中共有%d个元素' % len(stopword_set))

dataSetPath = './simplifyweibo_4_moods.csv'
testSplit = 0.2

# 读取并打乱数据
allData = pd.read_csv(dataSetPath).values[:5000]
allDataNum = allData.shape[0]

print(allData.shape)

# 划分训练集与测试集
trainSet = allData[:int(allDataNum * (1 - testSplit))]
testSet = allData[int(allDataNum * (1 - testSplit)):]

# 划分标签与特征值 , 并把标签转换为独热编码
xTrain = trainSet[:, 1]
yTrain = tf.one_hot(trainSet[:, 0], depth=4)
xTest = testSet[:, 1]
yTest = tf.one_hot(testSet[:, 0], depth=4)

cutWords_list = []
startTime = time.time()
content_series = xTrain
i = 0
for content in content_series:
    cutWords = [k for k in jieba.cut(content, True) if k not in stopword_set]
    if (i + 1) % 3000 == 0:
        usedTime = time.time() - startTime
        print('前%d篇文章分词共花费%.2f秒' % (i + 1, usedTime))
    i += 1
    cutWords_list.append(cutWords)
# print(cutWords_list)

startTime = time.time()
word2vec_model = Word2Vec(cutWords_list, size=200, iter=10, min_count=20)
usedTime = time.time() - startTime
print('形成word2vec模型共花费%.2f秒' % usedTime)

print(word2vec_model.wv.most_similar('家猫'))


def get_contentVector(cutWords, word2vec_model):
    vector_list = [
        word2vec_model.wv[k] for k in cutWords if k in word2vec_model
    ]
    contentVector = np.array(vector_list).mean(axis=0)
    return contentVector


startTime = time.time()
contentVector_list = []
for i in range(len(cutWords_list)):
    cutWords = cutWords_list[i]
    if (i + 1) % 3000 == 0:
        usedTime = time.time() - startTime
        print('前%d篇文章内容表示成向量共花费%.2f秒' % (i + 1, usedTime))
    contentVector_list.append(get_contentVector(cutWords, word2vec_model))
X = np.array(contentVector_list)
print(np.isnan(X).any(axis=1))
print('=' * 20)
print(X)