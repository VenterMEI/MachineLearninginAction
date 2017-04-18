from math import log
import operator


def createDataSet():
	dataset=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	attrilist=['no surfacing','flippers']		#'no surfacing'为dataset第一列属性的label，'flippers'为dataset第二列属性的label
	return dataset,attrilist



#计算香农熵(Shannon Entropy)，熵越高则混合的数据越多
def calShannonEnt(dataset):
	row=len(dataset)
	countdict={}
	for vector in dataset:
		label=vector[-1]
		countdict[label]=countdict.get(label,0)+1
	shannonent=0
	for key in countdict:
		prob=countdict[key]/row
		shannonent-=prob*log(prob,2)
	return shannonent



#按照给定特征划分数据集
def splitDataSet(dataset,axis,value):	#axis 用于划分数据集的特征，为dataset的列下标		#value 属性值
	retDataSet=[]
	for vector in dataset:
		if vector[axis]==value:
			reducedvector=vector[:axis]
			reducedvector.extend(vector[axis+1:])
			retDataSet.append(reducedvector)
	return retDataSet
'''
#在函数中传递的是列表的引用，所以在函数内部对dataset的修改会影响到该列表对象的整个生存期。为了避免这一影响，我们重新构造了一个retDataSet.

#extend()与append():
a=[1,2,3]
b=[4,5,6]
a.append(b)
print(a)	#[1,2,3,[4,5,6]]
a=[1,2,3]
a.extend(b)	
print(a)	#[1,2,3,4,5,6]
'''	



#选择最好的属性来划分:对dataset的每个属性都计算香农熵并比较信息增益。选择信息增益最大的一项，返回其属性的index。
#Feature特征   Attribut属性 两者的含义是相同的
def chooseBestFeatureToSplit(dataset):
	numFeature=len(dataset[0])-1		#dataset的属性数，为dataset的列数减一，因为labels项占据了dataset的一列
	baseEnt=calShannonEnt(dataset)	#原始的香农熵
	bestInfoGain=0					#最大信息增益
	bestFeature=-1
	for i in range (numFeature):
		featureList=[row[i] for row in dataset]	#等价于fetureList=dataset[:,i]，但此时的dataset需为numpy的array列表，而非list列表
		#这个列表推导式每次将dataset一行的第i项元素加入列表，整个dataset遍历完，featureList就接收到了由dataset第i列组成的列表
		featureSetList=set(featureList)		#set(list)返回一个列表，列表里的元素是对list求集合的结果，即去除重复值。
		newEnt=0
		for value in featureSetList:
			subdataset=splitDataSet(dataset,i,value)
			prob=len(subdataset)/len(dataset)
			newEnt+=prob*calShannonEnt(subdataset)	#熵的期望=子集的熵*子集占整个数据集的比重
		infoGain=baseEnt-newEnt				#熵越小越好，所以infoGain（信息增益）越大越好
		if infoGain>bestInfoGain:
			bestInfoGain=infoGain
			bestFeature=i
	return bestFeature



#输入为标签列表，输出为出现次数最多的标签
def majorityCnt(labellist):		#labellist是dataset或dataset子集的最后一列，标签列
	labelcount={}
	for label in labellist:
		labelcount[label]=labelcount.get(label,0)+1
	sortedlabelcount=sorted(labelcount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedlabelcount[0][0]



#构造决策树
#注意：原代码中第二个参数不是attrilist而是labels，但labels易于fish的'yes'和'no'标签混淆，故换成attrilist表示'no surfacing','flipper'这样的列属性
def createTree(dataset,attrilist):
	labellist=[row[-1] for row in dataset]				#labelslist 标签列表
	if labellist.count(labellist[0])==len(labellist):	#list.count(x)	计数x在列表list中出现的次数。	
		return labellist[0]				#递归结束条件一：这里如果'=='成立，表示该数据集中的所有label(fish)完全相同，无需继续划分数据集,直接返回该标签。
	if len(dataset[0])==1:
		return majorityCnt(labellist)	#递归结束条件二：使用完了所有特征，仍不能将数据集划分成一致的label，返回出现次数最多的标签代表这个分组
	bestFeature=chooseBestFeatureToSplit(dataset)		#找到香农熵最小的那一个属性的index
	bestFeatureLabel=attrilist[bestFeature]		
	myTree={bestFeatureLabel:{}}		#myTree是一个字典，字典的键是属性，字典的值是另一个字典
	subattrilist=attrilist[:]		#列表作为参数传递的是列表的引用，故构造subattribute防止attributelist改变
	del(subattrilist[bestFeature])			#del(list[index]):列表list的删除函数，index就是attrilist要删掉的内容。
	featureValues=[row[bestFeature] for row in dataset]		#featureValues为bestfeature那项属性的所有属性值的列表
	featureValuesSet=set(featureValues)	#featureValuesSet为bestfeature那项属性可能出现的属性值的集合
	for value in featureValuesSet:
		myTree[bestFeatureLabel][value]=createTree(splitDataSet(dataset,bestFeature,value),subattrilist)	
		#createTree可能返回label作为值，也可能继续返回嵌套的字典作为值
		#{属性：{属性值种类1：可能返回一个标签，属性值种类二：可能返回一个嵌套字典}}
	return myTree	

#createTree和plotTree只是构造了决策树并将其可视化，但对输入的向量进行分类返回其label就要交给classify
#比如已经求得决策树，那么对[1,1,label?],其vector=[1,1],那么它的label是多少就需要借助classify函数。
def classify(tree,attrilist,vector):
	firstattri=list(tree.keys())[0]
	index=attrilist.index(firstattri)
	key=vector[index]
	value=tree[firstattri][key]
	if type(value).__name__=='dict':
		return classify(value,attrilist,vector)
	else:
		return value
'''
isinstance(object,type)函数：
>>> isinstance(1, int)
True
>>> isinstance(1.0, float)
True
>>>isinstance(1,(int,float))
True
注意：isinstance与type区别：type()的话类型必须一致，而在isinstance()中object可以与type一致，也可以是type的子类
'''			


#由于计算决策树的开销很大，所以使用pickle模块序列化对象，在需要的时候可以再读取出来。任何对象都可以序列化，字典也不例外。
def storeTree(inputTree,filename):
	import pickle
	fw = open(filename,'wb')
	pickle.dump(inputTree,fw)
	fw.close() 
def grabTree(filename):
	import pickle
	fr = open(filename,'rb')
	return pickle.load(fr)
'''
pickle模块：
如字典，列表以及自定义的结构的对象，程序关闭后消失，再使用就要重新构造。
pickle可以将这些对象存储在文件中，需要时直接读取，并且可以被识别还原成对象，而不用重新构造。
首先要存储，打开文件，用dump函数写入。
然后读取，打开文件，用load函数加载到程序里。
注意，读和写都是二进制。
'''



'''
dataset,attrilist=createDataSet()
print(dataset)
print(calShannonEnt(dataset))		
bestFeature=chooseBestFeatureToSplit(dataset)
print(bestFeature)
mytree=createTree(dataset,attrilist)
print(mytree)
print(classify(mytree,attrilist,[1,0]))
storeTree(mytree,'fish')
treeoffish=grabTree('fish')
print(treeoffish)
'''
		