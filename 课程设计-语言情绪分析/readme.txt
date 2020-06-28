项目结构：
    checkpoint                          # 保存了每一轮训练的权重，用来连续训练，计算能力低，所以在原有的 参数下继续训练，文件数量多了以后，可以手动删除，但请保留最近几次权重，按时间排序，删除生成久的
    logs                                # 保存了每轮 训练集和验证集 的 损失与准确率，可以使用tensorboard 查看，文件数量多了以后，可以手动删除
    model                               # 整个模型的保存位置，在应用时，加载整个模型来应用
    predict.py                          # 创建单词字典
    simplifyweibo_4_moods.csv           # 数据集文件 详细参考： https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb
    tmp.py                              # 功能测试，没用
    train_model.py                      # 训练模型
    word_create.py                      # 单词字典创建
    word_dict.npy                       # 生成的单词字典，保存了单词在数据集中出现的次数, 全部的80% 生成的 4万 个字汇
    word_dict_10w.npy                   # 全部样本，生成的 10万 个词汇
    word_dict_20w.npy                   # 全部样本，生成的 20万 个词汇
    x_train.npy                         # 转换为词向量后训练样本
    x_test.npy                          # 转换为词向量后测试样本
    yTrain.npy                          # 训练样本的标签，独热编码形式
    yTest.npy                           # 测试样本的标签，独热编码形式
    stopwords.txt                       # 常用的停顿词， 这个还没有开始使用