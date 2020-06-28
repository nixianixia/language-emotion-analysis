import jieba
import jieba.analyse
import tensorflow as tf
import numpy as np
import json
'''
    这个文件做一些功能测试，没什么用
'''
text = ["我是一个重庆师范大学的在读学生", '把句子中所有的可以成词的词语都扫描出来', '对长词再次切分，提高召回率，适合用于搜索引擎分词']
# wl = []
# for t in text:
#     wl += list(jieba.cut(t))

# print(list(wl))

mytext = '''
回覆声援一下！就没有一个人肯说句公道话，就没有一家媒体肯问问这是怎么回事。任其恶性发展蔓延······当我见到网络和各种媒体上，包括戏子、主持人和商人们在扩大传播着冒名仓央嘉措的烂诗，我的悲愤非常具体。谁说不是呢一个傍晚，我徘徊在布达拉宫前面的广场上······而这座模仿天安门的广场另一边，同样有一个高耸的纪念碑，只是旁边多了一个现在很多城市都热衷建立的音乐喷泉·····暮然回首，我看见对面雄伟而神秘的布达拉宫，正傲然注视着这个喧闹的俗世，突然一种莫名的悲愤和无奈涌向心头。
'''

# mytext = "what fuck it doesn't work fuck ,fuck you"

tags = jieba.analyse.extract_tags(mytext, topK=80)
index = range(1, len(tags) + 1)
zp = zip(tags, index)
dicttags = dict(zp)
print(dicttags)
print('=' * 80)
wl = list(jieba.cut(mytext))
print(len(wl))
# print('=' * 80)
# wl = list(jieba.cut_for_search(mytext))
# print(wl)
# nt = jieba.analyse.analyzer.Tokenizer()
