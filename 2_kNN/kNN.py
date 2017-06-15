from numpy import * 
import operator
from os import listdir

def createDataSet():
	group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels


#kNN算法：计算点inX与dataset中各点距离并按距离排序。统计前k个点中出现频数最高的标签即为inX的预测标签。
def classify0(inX , dataset , labels , k):
	#计算dataset各点到inX的距离
	datasetsize=dataset.shape[0]		#np.shape[0]表示返回矩阵行数,group.shape[0]=4
	diffMat=tile(inX,(datasetsize,1))-dataset 	#np.tile(A,reps)用来扩增矩阵。diffMat=array([[-1,-1.1],[-1,-1],[0,0],[0,-0.1]])
	sqdiffMat=diffMat**2					#sqdiffMat=array([[1,1.21],[1,1],[0,0],[0,0.01]])
	sqdistances=sqdiffMat.sum(axis=1)	#np.sum()矩阵求和。		
	distances=sqdistances**0.5
	#对距离最小的k个点的标签进行统计
	sorteddistindex=distances.argsort()	#np.argsort()按照array()中值的大小从小到大将索引排序，并以array()的形式返回。
	classcount={}	#创建一个字典用于对分类计数	
	for i in range(k):
		label=labels[sorteddistindex[i]]
#		from collections import defaultdict 		
#		classcount.setdefault(label,0)	#当label这个键不存在时，创建这个键，并将其值设为0
		classcount[label]=classcount.get(label,0)+1		#dict.get(key[,default]):当键存在时返回键的值，当键不存在时返回default值。
	#排序找到最高频的标签并返回	
	sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)		#items()以view的方式返回字典的键值对
	return sortedclasscount[0][0]
#对测试集([0,0])进行训练,结果返回'B'。
#group,labels=createDataSet()	
#print(classify0([0,0],group,labels,3))	
'''
#np.tile()	扩增矩阵
tile(A,reps),在classify0中，A=inX=[0,0],reps=(4,1)
reps=(4,1)表示产生一个[A',A',A',A'],其中A'=[A*1]。这里4指4个A',1指A'是A的n个拷贝数。故tile产生array([[0,0],[0,0],[0,0],[0,0]])
更多例子：
print(np.tile(1.3,3))
print(np.tile((1,2,3),2))
a=[[1,2,3],[4,5,5]]
print(np.tile(a,2))
print(np.tile([1,2,3],[2,3,2]))

#np.sum()	矩阵求和
a=np.sum([[0,1,2],[2,1,3]])				#9					#无axis表示矩阵内所有元素之和
a=np.sum([[0,1,2],[2,1,3]],axis=0)  	#array([2,2,5])		#axis=0表示逐列求和
a=np.sum([[0,1,2],[2,1,3]],axis=1)  	#array([3,6])		#axis=1表示逐行求和
另外，min与max函数与sum类似
max()是全矩阵的最大值。max(0)返回矩阵每列最大值的array，max(1)返回矩阵每行最大值的array

#np.argsort()	索引排序
一维数组的排序
x = np.array([8, 3, 5])
np.argsort(x)						#array([1, 2, 0])
二维数组的排序
x = np.array([[0, 3], [2, 2]])
np.argsort(x, axis=0) #按列排序		#array([[0, 1],[1, 0]])
np.argsort(x, axis=1) #按行排序		# array([[0, 1],[0, 1]])

#dict.items()：以view的方式返回字典的键值对
在python3.x中，
Remove dict.iteritems(), dict.iterkeys(), and dict.itervalues(). 
Instead: use dict.items(), dict.keys(), and dict.values() respectively.
And dict.keys(), dict.items() and dict.values() return “views” instead of lists.

#operator.itemgetter(num)：
如在上例中，字典classcount每次返回一个键值对给key作为参数。itemgetter()函数则会对传入的参数（键值对）进行处理：
当num为0时，则取传入参数的第一项作为返回供sorted函数作为排序的key;当num为1时，则取传入参数的第二项。
'''


#约会网站配对
#将date的文本记录转化为可以直接用于classify0函数训练的训练样本矩阵和类标签向量
def file2matrix(url):
	file=open(url)
	lines=file.readlines()	#read()将文本一次读取为一个string readline()每次读取一行 readlines()返回一个列表，列表的元素是文本的每一行
	mat=zeros([len(lines),3])	#np.zeros()创建一个所有值均为0的矩阵，通过zeros内部的参数的确定矩阵的大小和维数，如果构造多维矩阵就传入数组
	labels=[]
	index=0
	for line in lines:
		line=line.strip()				#删除首尾的空白符（包括'\n', '\r',  '\t',  ' ')
		listfromline=line.split('\t')	#返回一个列表 	#split()无参数时，默认切割空字符，包括字母数字下划线
		mat[index]=listfromline[0:3]
		labels.append(int(listfromline[3]))		#强制类型转换
		index+=1
	return mat,labels
'''
#s.strip(rm):返回一个新字符串s’，s’是在s的基础上去除s的首尾出现在rm中的字符
s为字符串，rm为要删除的字符序列
s.strip(rm)        删除s字符串中开头、结尾处，出现在rm删除序列的字符
s.lstrip(rm)       删除s字符串中开头处，位于 rm删除序列的字符
s.rstrip(rm)      删除s字符串中结尾处，位于 rm删除序列的字符
当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')		
a = '     123'
a.strip()			#a='123'
当rm不为空，先从开头开始逐个读取string的一个个字符，如果出现在rm中就删除，一旦未出现在rm中就结束。再从结尾处遍历一遍。
a = '123abc'
b = '1213a2bc'
a.strip('12')		#返'3abc'
a.strip('21')		#'3abc'
b.strip('12')		#'13abc'
注意，从意义上来说，rm不是一个字符串而是一个删除字符的集合。rm内部字符的排列顺序本身并不影响删除的结果。
其次，strip不改变字符串本身，而是将修改创建在一个新字符串上。
'''



#数据的归一化：将所有数据都归一化到0-1之间—————>newvalue=(oldvalue-minvalue)/(max-min)
def autonorm(dataset):
	minvalues=dataset.min(0)		#min(0)矩阵的每列最大值。
	maxvalues=dataset.max(0)		#max(1)矩阵的每行最大值
	ranges=maxvalues-minvalues
#	normdata=zeros(shape(dataset))	#shape(dataset)返回一个元祖，第一项是dataset的行数，第二项是列数，zeros再以此构造相同大小的全0矩阵。	
	row=dataset.shape[0]		#shape[0]返回dataset的行数
	#num_of_row,num _of_cloumn=shape(dataset)
	dataset=dataset-tile(minvalues,(row,1))		#np.tile(A,reps)用来扩增矩阵。这一行为oldvalue-minvalue
	dataset=dataset/tile(ranges,(row,1))			#这一行再除以（max-min），至此，数据集归一化完成。
	return dataset,ranges,minvalues



#测试分类器的准确率
def datingclasstest():
	hoRatio=0.1
	mat,labels=file2matrix('D:\ADA\save\python\MachineLearninginAction\Ch02\datingTestSet2.txt')
	mat,ranges,minvalues=autonorm(mat)		#没有标准化，错误率达到0.24
	numcolumn=mat.shape[0]
	numtest=int(numcolumn*hoRatio)
	errorcount=0
	for i in range(numtest):
		classifierlabel=classify0(mat[i],mat[numtest:numcolumn],labels[numtest:numcolumn],3)	#用后90%的训练集训练前10%的数据集的label
		print('the clssfier came back with {}, the real answer is {}'.format(classifierlabel,labels[i]))
		if classifierlabel!=labels[i]:
			errorcount+=1
	print('the error count is {}.the total error rate is {}'.format(errorcount,errorcount/numtest))
	return	


#配对
def classifyperson():
	result=['not at all','in small doses','in large doses']
	percenttats=float(input('percent of time you spend playing vedio games:'))
	ffmiles=float(input('frequent flier miles:'))
	icecream=float(input('liters of ice cream:'))
	mat,labels=file2matrix('D:\ADA\save\python\MachineLearninginAction\machinelearninginaction\Ch02\datingTestSet2.txt')
	mat,ranges,minvalues=autonorm(mat)
	inX=array([ffmiles,percenttats,icecream])
	print(minvalues)
	classifierlabel=classify0((inX-minvalues)/ranges,mat,labels,3)
	print('you will probably like this person:{}'.format(result[classifierlabel-1]))
	return
#classifyperson()



#手写识别系统
#将图像转化为1维矩阵
def img2vector(url):
	vector=zeros([1,1024])
	file=open(url)
	for i in range(32):
		line=file.readline()
		for j in range(32):
			vector[0,i*32+j]=int(line[j])
	return vector
vector=img2vector(r'D:\ADA\standardCodeAndSamples\MachineLearninginAction\Ch02\recognize\trainingDigits\0_13.txt')


#测试kNN算法作用于手写识别上的错误率
def handwritingClassTest():
	#将training目录下的所有文件转化为label和mat
    hwLabels = []
    trainingFileList = listdir('trainingDigits')    	#os.listdir(url)	返回一个列表，列表内容是url这个目录里的所有文件名
    numfile = len(trainingFileList)
    trainingMat = zeros((numfile,1024))					#创建一个0矩阵，每行1024列，有多少个文件就有多少行
    for i in range(numfile):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     		#take off .txt
        classNumStr = int(fileStr.split('_')[0])		#take off _index
        hwLabels.append(classNumStr)					#标签
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)	#矩阵
    #将test目录下的所有文件逐个用training的数据去训练，得出错误率
    testFileList = listdir('testDigits')        		
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))


