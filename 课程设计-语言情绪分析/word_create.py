'''
    文件描述
    主要用来生成单词字典
    单词字典 在 将文本转为词向量时 需要使用

    性能优化：
        这种列表全部连接后，在去分词的方法，比较暴力，没有去寻找其它方法

    优化方式
        去除特殊符号
        去除停顿词
        用搜索模式来建立更多有关联的词
        放大 或 缩小 高频词数量         注意调整模型的 嵌入层 的相关参数
'''
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import jieba
import jieba.analyse

dataSetPath = './simplifyweibo_4_moods.csv'
testSplit = 0.0

# 读取并打乱数据
pdAll = pd.read_csv(dataSetPath).values
allData = np.vstack(
    (pdAll[0:30000], pdAll[200000:207500], pdAll[208000:213000],
     pdAll[214000:220000], pdAll[221000:233000], pdAll[256000:280000],
     pdAll[300000:337000], pdAll[338000:345000]))
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

print("拼接中")
mytext = ''
for t in xTrain:
    mytext += t.strip().replace('\r', '').replace('\n', '') + '\n'


def create_wordDict(text, maxnum=40000):
    print('拼接完毕\n分词中')
    tags = jieba.analyse.extract_tags(text, topK=maxnum)
    index = range(1, len(tags) + 1)
    zp = zip(tags, index)
    dictTags = dict(zp)
    print('分词完毕\n保存中')
    np.save("word_dict_" + str(int(maxnum / 10000)) + "w.npy", dictTags)
    print(dictTags)
    print("单词字典创建完成")


create_wordDict(mytext, 40000)
# create_wordDict(mytext, 100000)
