'''
    这是模型的应用模块
    会从本地加载训练过的权重文件然后进行预测
    加载不到权重文件，请执行 train_model.py  进行模型的训练
'''

import tensorflow as tf
import numpy as np
import jieba
# 模型路径
modelPath = './model/'
# 标签映射
moods = {0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}
# 加载模型
model = tf.keras.models.load_model(modelPath)
# 加载单词字典
wordDict = np.load('word_dict.npy', allow_pickle=True).item()


def texts_to_sequences(xTrain, maxlen=100):
    sequences = []
    for text in xTrain:
        wordList = list(jieba.cut(text))
        sequence = []
        for word in wordList:
            if len(sequence) < maxlen and word in wordDict.keys():
                sequence.append(wordDict[word])
        sequence += [0] * (maxlen - len(sequence))
        sequences.append(sequence)
    return tf.constant(sequences)


def displayTextMood(text):
    print(text)
    inputSeq = texts_to_sequences([text], maxlen=100)
    pred = model.predict(inputSeq)
    print('预测值：', moods[np.argmax(pred)])


displayTextMood(
    "现场报道，很直观…我家附近啊，还好今天父母都不在家危险无处不在，我都经常路过那段昆明东风路云南饭店与艺术剧院现场。因拆迁导致墙体倒塌，路人死伤若干，具体数字不详"
)
