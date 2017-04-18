'''
p(A|B)=p(B|A)P(A)/P(B)	条件概率
P(Ci|W)=p(W|Ci)p(Ci)/p(W)		、
如果Ci是label，W看作将一个事物抽象而成的若干特征。那么P(Ci|W)就是这个有若干特征指代的事物属于Ci的概率。
那么预测事物W属于哪个标签，只需求出事物W在所有标签中的概率，将概率最高的那个作为标签即可。
因为P(C1|W)与P(C2|W)的比较等价于P(C1|W)/p(W)与P(C2|W)/p(W)的比较。故只需知道p(W|Ci)p(Ci)的大小在C1和C2中谁大即可预测w属于C1还是C2.
'''
from numpy import *
import feedparser

def loadDataSet():
	dataset=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
			 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
			 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
			 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
			 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
			 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec=[0,1,0,1,0,1]    #1代表其中有侮辱性的字眼，0表示没有
	return dataset,classVec

#将dataset中出现的所有词转换成集合，并返回字符串列表
def createVocabList(dataset):
	vocabSet=set([])		#创建集合
	for line in dataset:
		vocabSet=vocabSet|set(line)		#| 集合并操作。注意，这里不能直接set(dataset),那样集合的元素不是词而是列表
	return list(vocabSet)

#根据字符串列表将input转换成词向量，input中词若也在字符串列表中，则对应位置的0变为1.
def wordsVec(vocabList,inputline):
	returnVec=[0]*len(vocabList)		#创建一个长度和vocablist一致的一维全0列表
	for vocabulary in inputline:
		if vocabulary in vocabList:
			#returnVec[vocabList.index(vocabulary)]=1	#这个词在vocabList出现过，那么向量对应的位置就改为1.
			returnVec[vocabList.index(vocabulary)]+=1	#由词集模型(出现与否)转化为词袋模型(出现几次)
	return returnVec

#计算p1Vector,p0Vector和pAbusive，前两个分别是在分类1和0中各个词出现的权重
#参数1为一个由dataset转化而来的矩阵，矩阵的每一行是一个词向量，参数2即labels(classVec)，表示trainMat的每一行是否为侮辱性
def trainNB(trainMat,trainCategory):	#NB naive basyes	#trainMat和trainCategory均为array类型，而不是list类型
	numLine=len(trainMat)
	numVoca=len(trainMat[0])
	pAbusive=sum(trainCategory)/numLine		#侮辱性句子在所有句子中的频率
	p0Num=ones(numVoca)
	p1Num=ones(numVoca)	
	count1=2
	count0=2	 
	for i in range(numLine):
		if trainCategory[i]:
			p1Num+=trainMat[i]
			count1+=sum(trainMat[i])
		else:
			p0Num+=trainMat[i]
			count0+=sum(trainMat[i])		
	p1Vector=array(log(p1Num/count1))	#一个向量，向量的每列都代表一个词，其值为该词的频率(该词在label中出现频数/count1)
	p0Vector=array(log(p0Num/count0))	#list不可以和num或list做除法，array可以和array或num做除法
	return 	p0Vector,p1Vector,pAbusive
'''
p1Vector和p0Vector再取对数的操作，是为了避免多个很小的数相乘导致下溢或四舍五入的错误。
且In(a*b)=In(a)+In(b),这对于概率常见的连乘计算很有用
但是为了避免出现In(0)，p0Num和p1Num使用ones函数而非zeros函数。count0和count1也相应的由0改为2.
'''


#计算词向量出现在各分类中的概率，概率最大的分类作为返回
#参数1为待分类的词向量，参数2到4是trainNB函数返回的结果。
def classifyNB(vectorClassify,p0Vector,p1Vector,pClass1):
	p1=sum(vectorClassify*p1Vector)+pClass1		
	#sum()的结果为p(W|Ci)的对数，p(W|Ci)=p(W0,W1,W2...Wn|Ci)=每个词在分类Ci中出现的权重*这个词是否在W中出现。pClass1即为p(Ci)。
	#所以p1其实并不是p(Ci|W)的对数,而是p(CiW)的对数.即p1是Ci和W同时发生的概率的对数。当然，不除p(w)对比大小没有影响
	p0=sum(vectorClassify*p0Vector)+(1-pClass1)
	if p1>p0:
		return 1
	else:
		return 0


def testingNB():
	dataset,classVec=loadDataSet()
	vocabList=createVocabList(dataset)
	trainMat=[]
	for line in dataset:
		trainMat.append(wordsVec(vocabList,line))
	p1v,p0v,pAb=trainNB(trainMat,classVec)
	testline=['love','my','dalmation']
	testVec=wordsVec(vocabList,testline)
	print(classifyNB(testVec,p0v,p1v,pAb))
	testline=['garbage','dalmation','stupid']
	testVec=wordsVec(vocabList,testline)
	print(classifyNB(testVec,p0v,p1v,pAb))
	return

#testingNB()


#输入是一封邮件的文本内容，输出是一个字符列串表
def textParse(bigString):   	#parse 解析 
	import re
	stringList=re.split(r'\W*',bigString)		#split与正则，加r则\不用再转义。\W与\w含义相反，表示所有不是字母数字下划线的字符。*为数量符
	return [string.lower() for string in stringList if len(string)>2]		#列表表达式
	#每个字符串都要大于2的原因。第一，字符长度小于等于2的单词没有什么含义，如is,am。第二，类似由'M.L.'分割成'M'，'L'，这种词同样没有价值。
	#最后，split可能会切出部分空字符。如'M.L.'返回['M','L','']，但是，书中的'M.L. I'返回是['M','L','I']而不是['M','L','','I']


def spamTest():
	docList=[]		#二维矩阵，每一行记录一条邮件的字符串列表
	classList=[]	#记录分类信息
	fullList=[]		#记录邮件中出现的所有词
	from os import listdir
	hamemails=listdir('D:\ADA\save\python\MachineLearninginAction\Ch04\email\ham')
	spamemails=listdir('D:\ADA\save\python\MachineLearninginAction\Ch04\email\spam')	#spam为垃圾邮件
	for url in hamemails:
		vocabList=textParse(open('D:\ADA\save\python\MachineLearninginAction\Ch04\email\ham\{}'.format(url)).read())
		docList.append(vocabList)		#append与extend区别
		fullList.extend(vocabList)
		classList.append(0)
	for url in spamemails:
		vocabList=textParse(open('D:\ADA\save\python\MachineLearninginAction\Ch04\email\spam\{}'.format(url)).read())
		docList.append(vocabList)
		fullList.extend(vocabList)
		classList.append(1)	
	vocabSet=createVocabList(docList)	#遍历所有字符串，返回字符串的集合的列表
	#构造训练集和测试集
	trainingSetIndex=list(range(len(docList)))	#注意，一定要转换为list，range类型无法继续del操作。
	testSetIndex=[]			#trainingSetIndex和testSetIndex记录的是要选哪几个词做训练集和测试集，记录的是词在词列表中的下标。
	for i in range(int(len(docList)/5)):
		index=int(random.uniform(0,len(trainingSetIndex)))	#random.uniform(start,end) 产生一个范围在start到end之间的随机数
		testSetIndex.append(trainingSetIndex[index])
		del(trainingSetIndex[index])
	trainMat=[] ;trainClass=[]
	for i in trainingSetIndex:
		trainMat.append(wordsVec(vocabSet,docList[i]))
		trainClass.append(classList[i])
	p0v,p1v,pSpam=trainNB(trainMat,trainClass)
	errorCount=0
	for  i in testSetIndex:
		testVector=wordsVec(vocabSet,docList[i])
		if classifyNB(testVector,p0v,p1v,pSpam)!=classList[i]:
			errorCount+=1
	print(errorCount/len(testSetIndex))		
	return
#spamTest()

#从个人广告获取区域倾向
#统计最高频的三十个词
def calcMostFreq(vocabList,fullText):
	from operator import itemgetter
	worddict={}
	for word in fullText:
		worddict[word]=fullText.count(word)		#list的count(element)
	sortedList=sorted(worddict.items(),key=itemgetter(1),reverse=True)	#将词列表（不是词向量）按词频排序.
	#itemgetter(1)在这里表示worddict.items()返回的第二个值，即为字典的值
	return sortedList[:30]

#输入为两个RSS源，分别为两个地区的人发布的征婚广告。输出为分类的错误率统计
def localWord(feed1,feed0):
	docList=[];classList=[];fullText=[]
	#fullText用一个一维列表来记录所有词，便于后面calcMostFreq函数里统计高频词
	minlen=min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minlen):
		wordlist=textParse(feed1['entries'][i]['summary'])
		docList.append(wordlist)
		fullText.extend(wordlist)
		classList.append(1)
		wordlist=textParse(feed0['entries'][i]['summary'])
		docList.append(wordlist)
		fullText.extend(wordlist)
		classList.append(0)
	vocabList=createVocabList(docList)
	#去掉高频词
	top30words=calcMostFreq(vocabList,fullText)
	for word in top30words:		#若top30words是字典，那么这里的word就是键。但是注意，在calcMostFreq中，worddict的items()是view类型，已经不再是字典。
		vocabList.remove(word[0])	#remove只能用于list，del函数则list和array都适用
	#划分训练集和测试集
	trainingSetIndex=list(range(2*minlen));testSetIndex=[]
	for i in range(int(len(docList)/5)):
		index=int(random.uniform(0,len(trainingSetIndex)))	#random.uniform(start,end) 产生一个范围在start到end之间的随机数
		testSetIndex.append(trainingSetIndex[index])
		del(trainingSetIndex[index])
	trainMat=[] ;trainClass=[]
	for i in trainingSetIndex:
		trainMat.append(wordsVec(vocabList,docList[i]))
		trainClass.append(classList[i])
	p0v,p1v,pSpam=trainNB(trainMat,trainClass)		#这里pSpam代指feed1占标签的总数
	errorCount=0
	for  i in testSetIndex:
		testVector=wordsVec(vocabList,docList[i])
		if classifyNB(testVector,p0v,p1v,pSpam)!=classList[i]:
			errorCount+=1
	print(errorCount/len(testSetIndex))	
	return vocabList,p0v,p1v

#显示具有代表性的词汇显示
def getTopWord(ny,sf):
	from operator import itemgetter
	vocabList,p0v,p1v=localWord(ny,sf)
	topNY=[];topSF=[]
	#p>-6时词特别多，其实可以再加一层，对排序后的词只打印10个。这样特征词汇少一点。
	for i in range(len(p0v)):
		if p0v[i]>-4.8:
			topSF.append((vocabList[i],p0v[i]))		#每次插入一个元组，元组由这个词和该词的权重组成
		if p1v[i]>-4.8:
			topNY.append((vocabList[i],p1v[i]))		
	print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
	sortedSF=sorted(topSF,key=itemgetter(1))
	for item in sortedSF:
		print(item[0])	
	print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")	
	sortedNY=sorted(topNY,key=itemgetter(1))
	for item in sortedNY:
		print(item[0])	
	return	

ny=feedparser.parse('http://newyork.craigslist.org/stp/index/rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index/rss')
#vocabList,p0v,p1v=localWord(ny,sf)
getTopWord(ny,sf)

