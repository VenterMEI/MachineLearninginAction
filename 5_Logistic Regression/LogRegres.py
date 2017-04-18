from numpy import *



def loadData():
	fr=open(r'D:\ADA\save\python\MachineLearninginAction\Ch05\testSet.txt')
	dataMat=[];labelMat=[]
	for line in fr.readlines():
		lineList=line.strip().split()
		dataMat.append([1.0,float(lineList[0]),float(lineList[1])])		#整个1可以类似看做s=k1*y+k2*x+k3的k3
		labelMat.append(int(lineList[2]))	#int和上一行的float不能少，因为在强制类型转换之前，他们是字符串类型
	return dataMat,labelMat

#将0到1的阶跃函数转化为sigmoid函数，使得整个跳跃的过程易于处理
def sigmoid(inx):
	return 1/(1+exp(-inx))

#梯度上升
#方法：每次计算m个标签与通过weight计算的m个预测之差，差值*该点位置类似于梯度，将这个乘积再乘上学习率和weight相加，那么weight的预测就和真实的标签一致了
def gradeAscent(dataMat,labelMat):
	dataMat=mat(dataMat)	#numpy的矩阵类型，可以进行转置之类的矩阵操作
	labelMat=mat(labelMat).transpose()		#transpose()矩阵的转置函数
	m,n=shape(dataMat)
	alpha=0.001 			#学习速率，每轮学习对权重weight改变的大小
	cycle=500				#循环次数，决定学习多少次
	weight=ones([n,1])	
	for k in range(cycle):
		predictLabel=sigmoid(dataMat*weight)	#注意，整个预测的标签不是一个标签，而是一个m行1列的标签矩阵
		#预测标签=sigmoid(weight[0]*1+weight[1]*x+weight[2]*y).所以在loadData的dataMat.append()行，不一定非要插入1，插入2也行，weight[0]会对其进行修改。但是，不能插入0.
		error=labelMat-predictLabel
		weight=weight+alpha*dataMat.transpose()*error
	return weight

#随机梯度上升 		stochastic 随机
'''
和梯度上升不同，随机梯度上升每次只用一条数据而不是一组数据。因而dataMat不需要进行矩阵运算，error也只是一个值而非向量。
同样利用梯度，随机梯度上升更易于理解一点。
不同在于，梯度上升每运算一次就涉及到整个dataMat，梯度上升采用的是将error的m个差值直接合并作为梯度加到weight上。
而随机梯度上升则是每次只使用一条数据，立刻将error加到weight上。
必然，达到相同的回归效果，随机梯度上升算法的计算量必定远小于梯度上升
'''
def stocGradeAscent0(dataMat,labelMat):
	m,n=shape(dataMat)
	alpha=0.001
	weight=ones(n)
	cycle=800
	for j in range(cycle):
		for i in range(m):
			predictLabel=sigmoid(sum(dataMat[i]*weight))			#list和list无法相乘，array([a1,a2,a3])*array([b1,b2,b3])/[b1,b2,b3]=array([a1b1,a2b2,a3b3])
			error=labelMat[i]-predictLabel
	#		weight=weight+alpha*error*dataMat[i]			#error:list无法和浮点数相乘，list和int相乘是复制list。故出错，所以要将其改为array
			weight=weight+alpha*error*array(dataMat[i])
	return weight

#改进的随机梯度上升算法
def stocGradeAscent(dataMat,labelMat):
	m,n=shape(dataMat)
	dataMat=array(dataMat)
	weight=ones(n)
	cycle=1600
	indexcount=[0]*m 
	for i in range(cycle):
		indexList=list(range(m))
		for j in range(int(m*0.5)):			#为了体现出随机，我使用m*0.5.每轮学习只使用dataMat中一半的样本调整weight。否则所谓的随机只表现在学习的先后上。
			alpha=0.4/(1+i+j)+0.0005		#变化的学习率，先快后慢。常数项确保学习率不为0，每次都能学到东西		
			#index=int(random.uniform(0,len(indexList)))	#源码直接用index作为下标，这样并不随机，因为随着del函数不断执行，排在前面的数据出现的概率越来越大。
			randomindex=int(random.uniform(0,len(indexList)))	
			index=indexList[randomindex]
			indexcount[index]+=1
			predictLabel=sigmoid(sum(dataMat[index]*weight))
			error=labelMat[index]-predictLabel
			weight=weight+alpha*error*dataMat[index]
			del(indexList[randomindex])
	print(indexcount)		
	return weight		


#绘制分割线
def plotBestFit():
	import matplotlib.pyplot as plt
	dataMat,labelMat=loadData()
	m=shape(dataMat)[0]
	X1=[];Y1=[]
	X0=[];Y0=[]
	for i in range(m):
		if labelMat[i]==1:
			X1.append(dataMat[i][1]);Y1.append(dataMat[i][2])
		else:
			X0.append(dataMat[i][1]);Y0.append(dataMat[i][2])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.scatter(X1,Y1,c='green')
	ax.scatter(X0,Y0,c='red')
	#绘制分割线	
#	weight=gradeAscent(dataMat,labelMat)
	weight=stocGradeAscent0(dataMat,labelMat)
	x=arange(-3,3,0.1)		#arange(start,end,step)函数,返回一个array数组，数组元素再start到end之间(左闭→右边)，间隔为step
	y=(-float(weight[0])-float(weight[1])*x)/float(weight[2])		
#	y=(-weight[0]-weight[1]*x)/weight[2]		#元素代码会报错
	ax.plot(x,y)
	plt.show()
	return
#plotBestFit()	
'''
首先，ones([10])和ones([1,10])不一样
print(ones(10))				#[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
print(zeros([1,10]))		#[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
print(ones([10]))			#[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
在此基础上，再看看书中源码的问题。
注意：在gradeAscent和ployBestFit函数中，weight的值为
[[ 4.12414349]
 [ 0.48007329]
 [-0.6168482 ]]
weight是一个n行1列的array数组，其中n与dataMat中的列数一致。
对于书中y=(-weight[0]-weight[1])*x）/weight[2]代码，则y的值形如[[，，，]].
那么ax.plot(x,y)中x是一个1行m列的array数组，而y是一个仅有一个元素的array数组，只不过该元素是一个1行m列的array数组。所以报错。
y之所以形如[[，，，]]，是因为weight[1]*x时，是由于[[0.48]]*[，，，],这相当于[[0.48]*[，，，]]
为了避免出现这样的情况，将weight[1]强制类型转换。
此外，weight[0]+[，，，]形如[[4.12]]+[，，，],这相当于[[4.12]+[，，，]].这也同样需要强制转换


另一个需要说明的是，weight=[4.12414349	0.48007329	-0.6168482]，在图中类似[0,0]的点和分割线差很远。
但是，在sigmid函数的处理下，点[0,0]的预测值为0.984，和标签1已经非常接近了，故不会导致weight的大幅变动。（其实，sigmid(4)=0.982,输入大于4就很接近1了）
反而是在分割线附近的点如[0.5,5.5]预测值为0.705，和标签差距较大。
'''
def errorcount():
	dataMat,labelMat=loadData()
	count=0
	for j in range(10):
		weight=stocGradeAscent(dataMat, labelMat)
		for i in range(len(labelMat)):
			value=sigmoid(sum(dataMat[i]*weight))
			if value>0.5:
				predictLabel=1
			else:	
				predictLabel=0
			if predictLabel != labelMat[i]:
				count+=1
	return count
#print(errorcount())
'''
源码中stocGradeAscent错误率比stocGradeAscent0高，是因为学习率过大,最后一轮中alpha=4/(1+150+1)+0.0001=0.0264,比stocGradeAscent0的0.001.
故将4/(1+i+j)改为0.4，这样在最后一轮循环，i=cycle=1600时，学习率alpha=0.00075.则与stocGradeAscent0的0.001大致相当.
学习率设置过大的问题是会将weight调过了。
另外，将stocGradeAscent的cycle设为stocGradeAscent0的双倍，这样两者才使用了相同数量的instance（由于stocGradeAscent中的m*0.5），这样比较才公平。
此外，errorcount在3左右是正常的，因为看数据的发布，就是没法一条线将数据完美分割。
'''