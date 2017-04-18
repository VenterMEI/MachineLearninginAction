from numpy import *

def loadData(fileName):
	fr=open(fileName)
	dataMat=[];labelMat=[]
	for line in fr.readlines():
		line=line.strip().split('\t')
		dataMat.append([float(line[0]),float(line[1])])
		labelMat.append(int(line[2]))
	return dataMat,labelMat

#为alpha i选择另一个alpha j与其配对
def selectJrand(i,m):
	j=i
	while j==i:
		j=int(random.uniform(0,m))
	return j

#为alpha设置上下界
def clipAlpha(aj,H,L):
	if aj>H:
		aj=H
	if aj<L:
		aj=L
	return aj

dataMat,labelMat=loadData(r'D:\ADA\save\python\MachineLearninginAction\Ch06\testSet.txt')	
print(mat(labelMat).transpose())