import re
from collections import Counter#统计字符出现次数
import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models import Word2Vec


model=Word2Vec.load('new.model')


# 对文章进行分句
def splitdata(text):
    text2 = (text.replace("\n", "").replace("。", "*").replace("！", "*")).split("*") #按整句划分
#    text2 = (text.replace("\n", "").replace("。", "*").replace("，", "*").replace("！", "*")).split("*") #按短句划分
    text2 = text2[:-1]
    return text2


# 取出句内非法字符
def cleandata_(text):
    return ''.join(a for a in re.findall('\w+', text))


# a为权重
def comWeight(text, a=0.01):  # text传入原文即可，文章会进行处理
    sentence = cleandata_(text)
    splitword = list(jieba.cut(sentence))
    lenth = len(splitword)
    count = Counter(splitword)
    for w in count.keys():
        count[w] = a / (a + (count[w] / lenth))
    return count


# 输入句子列表,每个句子要被清理过的
def comsenEmbedding(text, weights):
    senEmbedding = np.zeros([len(text), 400], dtype=np.float64)
    for i in range(len(text)):
        sen = text[i]
        wordlist = list(jieba.cut(sen))
        vec = np.zeros([1, 400], dtype=np.float64)
        numofZero = 0
        for j in range(len(wordlist)):
            if (wordlist[j] in model):
                vec += weights[wordlist[j]] * model[wordlist[j]]

            else:
                numofZero += 1
        #        senEmbedding[i] = vec/((len(wordlist)-numofZero))
        senEmbedding[i] = vec / ((len(wordlist) - numofZero) + 0.00001)
    return senEmbedding


# pca降到1维
def pca(vec):
    a = PCA(n_components=1)
    u = a.fit_transform(vec)
    return u


def get_final_senEmbedding(senEmbedding, u):
    senEmbedding = senEmbedding - u @ u.transpose() @ senEmbedding
    return senEmbedding


def get_Vec(text, weightes):  # 标题和全文
    text = cleandata_(text)
    textVec = np.zeros([1, 400], dtype=np.float64)
    wordlist = list(jieba.cut(text))
    numofZero = 0
    for j in range(len(wordlist)):
        if (wordlist[j] in model):
            textVec += (weightes[wordlist[j]] * model[wordlist[j]])
        else:
            numofZero += 1
    textVec = textVec / (len(wordlist) - numofZero)
    u = pca(textVec)
    textVec = textVec - u @ u.transpose() @ textVec
    return textVec


# 计算余弦值
def cosValue(a, b):
    cosValue = np.float64(a @ b.transpose()) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cosValue


# 计算相似性，默认a偏向于全文
def comRelevant(sen_vec, text_vec, title_vec, a=0.7):
    relevant = np.zeros([sen_vec.shape[0], 1], dtype=np.float64)
    for i in range(sen_vec.shape[0]):
        relevant[i] = a * (cosValue(sen_vec[i], text_vec)) + (1 - a) * (cosValue(sen_vec[i], title_vec))
    return relevant.flatten()


def knn_smooth(relevant):
    relevant_s = relevant.copy()
    for i in range(len(relevant)):
        if i == 0:
            relevant_s[i] = (relevant[i] + relevant[i + 1]) / 2
        elif i == len(relevant) - 1:
            relevant_s[i] = (relevant[i] + relevant[i - 1]) / 2
        else:
            relevant_s[i] = (relevant[i + 1] + relevant[i] + relevant[i - 1]) / 3
    return relevant_s


def get_index(relevant_s):
    argsort_relevant = np.argsort(relevant_s)  # 返回数组值从小到大的索引值
    # 平滑后的相关性索引
	# 从大到小
    index = argsort_relevant[::-1]  
    return index


def summarize(content,title,summary_pct=0.4):
    sentence = splitdata(content)
    weight = comWeight(content)
    senVec = comsenEmbedding(sentence,weight)
    u = pca(senVec)
    senVec = get_final_senEmbedding(senVec,u)
    titleVec = get_Vec(title,weight)
    contentVec = get_Vec(content,weight)
    rel = comRelevant(senVec,contentVec,titleVec)
    rel_knn = knn_smooth(rel)
    index=get_index(rel_knn)
    outnum = int(np.ceil(senVec.shape[0]*summary_pct))
    index2 = index[:outnum]
    #正则匹配标点
    sen_punc = []
    str = u'{}'.format(content)
    for i in range(senVec.shape[0]):
        #按content顺序输出
        if i in index2:
            pattern = re.compile(u'{}["，""。""！"]'.format(sentence[i]))
            if pattern.search(str):
                sen_punc_ = pattern.search(str).group()
                sen_punc.append(sen_punc_)
    summary = "".join(sen_punc)
    summary = summary.replace(summary[-1],"。")
    return summary


#filename = '/Users/han/Desktop/NLP/lesson 1/datasource-master/sqlResult_1558435.csv'
#articles = pd.read_csv(filename, encoding='gb18030')
#contents = articles['content'].tolist()
#titles = articles['title'].tolist()


#content = contents[31]
#title = titles[31]
print(summarize(content,title))