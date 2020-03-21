from gensim.test.utils import datapath
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re
import jieba
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#设置字体中文可见
plt.rcParams['font.sans-serif']=['Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False

#处理数据使其可传入PathLineSentences
file=open('/Users/han/PycharmProjects/NLP_project1/clean.txt')
content=file.read()
file.close()
articles=content
demo=re.sub('[a-zA-Z0-9]','',str(articles))
f=open('cleanwiki1.txt','w+')
for i in re.split(r'[，。\n\r]',demo):
    if i!='':
        data = list(jieba.cut(i, cut_all=False))
        data = re.findall('\w+', str(data))
        readline=''.join(j+' ' for j in data)+'\n'
        f.write(readline)
f.close()

#通过LineSentence处理语料可被Word2Vec运用
sentences1=LineSentence(datapath('/Users/han/PycharmProjects/NLP_project1/clean/cleanwiki1.txt'))
#size为词向量维度，400输出结果结果较好，但处理速度过慢
model=Word2Vec(sentences=sentences1,size=400,window=5,min_count=5)
#通过wiki语料库训练完模型后，使用新闻语料库继续训练模型
sentences2=LineSentence(datapath('/Users/han/PycharmProjects/NLP_project1/clean/cleannews1.txt'))
#得到第二个语料库的句子数量
for i,line in enumerate(sentences2):
    print(i)   #1997072
model.train(sentences2,total_examples=1997072,epochs=model.epochs)
#保存训练后的模型，方便下次直接调用而不需再进行上述训练过程
model.save('new.model')
#训练完模型后可直接调用
model=Word2Vec.load('new.model')
#测试词向量的语义相似性
result=model.most_similar(positive='美女')
for each in result:
    print(each[0],each[1])
#测试词向量的语义线性关系
def analogy(x1,x2,y1):
    result=model.most_similar(positive=[y1,x2],negative=[x1])
    return result[0][0]
print(analogy('中国','汉语','美国'))
print(analogy('美国','奥巴马','中国'))

#进行词向量的可视化时，上述模型需要时间过长，因此重新训练较简易模型
file=open('/Users/han/PycharmProjects/NLP_project1/clean/cleannews1.txt')
content=file.read()
file.close()
articles=content[1:1000000]   #随意调整大小，选择100000因为所需时间较少
f=open('cleannews2.txt','w+')
f.write(articles)
f.close()
sentences3=LineSentence(datapath('/Users/han/PycharmProjects/NLP_project1/cleannews2.txt'))
model1=Word2Vec(sentences=sentences3,size=100,window=5,min_count=50) #通过min_count控制单词个数
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels=[]
    tokens=[]
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    tsne_model=TSNE(perplexity=40,n_components=2,n_iter=2500,init='pca',random_state=23)
    new_values=tsne_model.fit_transform(tokens)
    x=[]
    y=[]
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16,16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],xy=(x[i],y[i]),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
    plt.show()
tsne_plot(model1)
