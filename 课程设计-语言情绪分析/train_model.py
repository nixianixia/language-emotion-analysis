import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
import jieba
import os
'''
    文件描述：
    主要用来训练模型

    优化方向
        调整模型，比如，字典长度，一句话的长度，更多神经网络层，增加单元数量，学习率等
        可以将单词用搜索模式来划分，寻找更多特征
        存储转为词向量后的数据，下次训练，直接加载，避免前面的数据处理的时间
'''
wordsum = 0
dataSetPath = './simplifyweibo_4_moods.csv'  # 数据集路径
wordDictPath = './word_dict_20w.npy'  # 使用不同词典要修改词嵌入层的input_dim, 这是一个左闭右开的区间，加个 一 万事大吉
testSplit = 0.03  # 测试集划分百分比


def texts_to_sequences(xTrain, maxlen=100):
    global wordsum
    sequences = []
    for text in xTrain:
        wordList = list(jieba.cut(text))
        sequence = []
        for word in wordList:
            if len(sequence) < maxlen and word in wordDict.keys():
                sequence.append(wordDict[word])
        wordsum += len(sequence)
        sequence += [0] * (maxlen - len(sequence))
        sequences.append(sequence)
    return tf.constant(sequences)


x_train = ''
x_test = ''
yTrain = ''
yTest = ''

if not os.path.isfile("x_train.npy"):

    # 读取并打乱数据
    allData = shuffle(pd.read_csv(dataSetPath).values)
    # pdAll = pd.read_csv(dataSetPath).values
    # allData = shuffle(
    #     np.vstack(
    #         (pdAll[0:30000], pdAll[200000:207500], pdAll[208000:213000],
    #          pdAll[214000:220000], pdAll[221000:233000], pdAll[256000:280000],
    #          pdAll[300000:337000], pdAll[338000:345000])))
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

    # 导入中文词典
    # 这个词典是提前分好的，因为需要很久
    wordDict = np.load(wordDictPath, allow_pickle=True).item()

    # 将句子转为词向量,并统一长度
    # 或许我们可以将这些词向量，全部保存到，方便训练
    x_train = texts_to_sequences(xTrain, maxlen=25)
    x_test = texts_to_sequences(xTest, maxlen=25)

    np.save("x_train.npy", x_train.numpy())
    np.save("x_test.npy", x_test.numpy())
    np.save("yTrain.npy", yTrain.numpy())
    np.save("yTest.npy", yTest.numpy())
else:
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    yTrain = np.load('yTrain.npy')
    yTest = np.load('yTest.npy')

print("x_train.shape", x_train.shape)
# print("平均单词数量", wordsum / allDataNum)
# 建立模型
model = tf.keras.models.Sequential()
# 嵌入
model.add(
    tf.keras.layers.Embedding(
        output_dim=32,  # 输出词向量的维度
        input_dim=200001,  # 输入的词汇表的长度
        input_length=25  # 输入Tensorr 的长度
    ))

# 平坦层
# model.add(tf.keras.layers.Flatten())

# 在使用 RNN 与 LSTM时 不需要平坦层
model.add(tf.keras.layers.LSTM(units=4))

# 全连接层
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
# 防止过拟合
model.add(tf.keras.layers.Dropout(0.3))

# 留着慢慢调吧
# model.add(tf.keras.layers.Dense(units=4, activation='relu'))

# model.add(tf.keras.layers.Dropout(0.3))

# 输出层
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# 设置模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 模型摘要
print(model.summary())

# 准备回调

logdir = './logs'  # 模型训练日志的保存路径
checkpoint_dir = './checkpoint/'  # 模型权重存放路径
checkpoint_path = checkpoint_dir + 'weight.{epoch:02d}-{val_loss:.2f}.H5'  # 检查点文件

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       save_weights_only=True,
                                       verbose=1,
                                       save_freq='epoch')
]

# 尝试加载模型权重,继续训练, 得到最新的检查点文件
model_filename = tf.train.latest_checkpoint(checkpoint_dir)
if model_filename != None:
    model.load_weights(model_filename)
    print("{}加载成功".format(model_filename))
else:
    print("未找到权重文件，需要重新训练")

# 训练模型
history = model.fit(x=x_train,
                    y=yTrain,
                    validation_split=0.02,
                    epochs=50,
                    batch_size=1000,
                    callbacks=callbacks,
                    verbose=2)

# 保存一个完整的模型用来应用
model.save("./model")

# 模型评估
testLoss, testAcc = model.evaluate(x_test, yTest, verbose=1)
print("损失: ", testLoss, "准确率：", testAcc)